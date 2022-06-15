use arrow::{datatypes::Schema, record_batch::RecordBatch};
use async_trait::async_trait;
use client_util::connection::{self, Connection};
use generated_types::ingester::IngesterQueryRequest;
use influxdb_iox_client::flight::{
    generated_types as proto,
    low_level::{Client as LowLevelFlightClient, LowLevelMessage, PerformQuery},
};
use observability_deps::tracing::debug;
use snafu::{ResultExt, Snafu};
use std::{
    collections::HashMap,
    fmt::Debug,
    ops::{Deref, DerefMut},
    sync::Arc,
};

pub use influxdb_iox_client::flight::Error as FlightError;

#[derive(Debug, Snafu)]
#[allow(missing_copy_implementations, missing_docs)]
pub enum Error {
    #[snafu(display("Failed to connect to ingester '{}': {}", ingester_address, source))]
    Connecting {
        ingester_address: String,
        source: connection::Error,
    },

    #[snafu(display("Failed ingester handshake '{}': {}", ingester_address, source))]
    Handshake {
        ingester_address: String,
        source: FlightError,
    },

    #[snafu(display("Internal error creating flight request : {}", source))]
    CreatingRequest {
        source: influxdb_iox_client::google::FieldViolation,
    },

    #[snafu(display("Failed to perform flight request: {}", source))]
    Flight { source: FlightError },

    #[snafu(display("Cannot find schema in flight response"))]
    SchemaMissing,
}

/// Abstract Flight client.
///
/// May use an internal connection pool.
#[async_trait]
pub trait FlightClient: Debug + Send + Sync + 'static {
    /// Send query to given ingester.
    async fn query(
        &self,
        ingester_address: Arc<str>,
        request: IngesterQueryRequest,
    ) -> Result<Box<dyn QueryData>, Error>;
}

/// Default [`FlightClient`] implementation that uses a real connection
#[derive(Debug, Default)]
pub struct FlightClientImpl {
    /// Cached connections
    /// key: ingester_address (e.g. "http://ingester-1:8082")
    /// value: CachedConnection
    ///
    /// Note: Use sync (parking_log) mutex because it is always held
    /// for a very short period of time, and any actual connection (and
    /// waiting) is done in CachedConnection
    connections: parking_lot::Mutex<HashMap<String, CachedConnection>>,
}

impl FlightClientImpl {
    /// Create new client.
    pub fn new() -> Self {
        Self::default()
    }

    /// Establish connection to given addr and perform handshake.
    async fn connect(&self, ingester_address: Arc<str>) -> Result<Connection, Error> {
        let cached_connection = {
            let mut connections = self.connections.lock();
            if let Some(cached_connection) = connections.get(ingester_address.as_ref()) {
                cached_connection.clone()
            } else {
                // need to make a new one;
                let cached_connection = CachedConnection::new(&ingester_address);
                connections.insert(ingester_address.to_string(), cached_connection.clone());
                cached_connection
            }
        };
        cached_connection.connect().await
    }
}

#[async_trait]
impl FlightClient for FlightClientImpl {
    async fn query(
        &self,
        ingester_addr: Arc<str>,
        request: IngesterQueryRequest,
    ) -> Result<Box<dyn QueryData>, Error> {
        let connection = self.connect(Arc::clone(&ingester_addr)).await?;

        let mut client = LowLevelFlightClient::<proto::IngesterQueryRequest>::new(connection);

        debug!(%ingester_addr, ?request, "Sending request to ingester");
        let request: proto::IngesterQueryRequest =
            request.try_into().context(CreatingRequestSnafu)?;

        let mut perform_query = client.perform_query(request).await.context(FlightSnafu)?;
        let (schema, app_metadata) = match perform_query.next().await.context(FlightSnafu)? {
            Some((LowLevelMessage::Schema(schema), app_metadata)) => (schema, app_metadata),
            _ => {
                return Err(Error::SchemaMissing);
            }
        };
        Ok(Box::new(PerformQueryAdapter {
            inner: perform_query,
            schema,
            app_metadata,
        }))
    }
}

/// Data that is returned by an ingester gRPC query.
///
/// This is mostly the same as [`PerformQuery`] but allows some easier mocking.
#[async_trait]
pub trait QueryData: Debug + Send + 'static {
    /// Returns the next `RecordBatch` available for this query, or `None` if
    /// there are no further results available.
    async fn next(&mut self) -> Result<Option<RecordBatch>, FlightError>;

    /// App metadata that was part of the response.
    fn app_metadata(&self) -> &proto::IngesterQueryResponseMetadata;

    /// Schema.
    fn schema(&self) -> Arc<Schema>;
}

#[async_trait]
impl<T> QueryData for Box<T>
where
    T: QueryData + ?Sized,
{
    async fn next(&mut self) -> Result<Option<RecordBatch>, FlightError> {
        self.deref_mut().next().await
    }

    fn app_metadata(&self) -> &proto::IngesterQueryResponseMetadata {
        self.deref().app_metadata()
    }

    fn schema(&self) -> Arc<Schema> {
        self.deref().schema()
    }
}

#[derive(Debug)]
struct PerformQueryAdapter {
    inner: PerformQuery<proto::IngesterQueryResponseMetadata>,
    app_metadata: proto::IngesterQueryResponseMetadata,
    schema: Arc<Schema>,
}

#[async_trait]
impl QueryData for PerformQueryAdapter {
    async fn next(&mut self) -> Result<Option<RecordBatch>, FlightError> {
        loop {
            match self.inner.next().await? {
                None => {
                    return Ok(None);
                }
                Some((LowLevelMessage::RecordBatch(batch), _)) => {
                    return Ok(Some(batch));
                }
                // ignore all other message types for now
                Some((LowLevelMessage::None | LowLevelMessage::Schema(_), _)) => (),
            }
        }
    }

    fn app_metadata(&self) -> &proto::IngesterQueryResponseMetadata {
        &self.app_metadata
    }

    fn schema(&self) -> Arc<Schema> {
        Arc::clone(&self.schema)
    }
}

#[derive(Debug, Clone)]
struct CachedConnection {
    ingester_address: Arc<str>,
    /// Real async mutex to
    maybe_connection: Arc<tokio::sync::Mutex<Option<Connection>>>,
}

impl CachedConnection {
    fn new(ingester_address: &Arc<str>) -> Self {
        Self {
            ingester_address: Arc::clone(ingester_address),
            maybe_connection: Arc::new(tokio::sync::Mutex::new(None)),
        }
    }

    /// Return the underlying connection, creating it if needed
    async fn connect(&self) -> Result<Connection, Error> {
        let mut maybe_connection = self.maybe_connection.lock().await;

        let ingester_address = self.ingester_address.as_ref();

        if let Some(connection) = maybe_connection.as_ref() {
            debug!(%ingester_address, "Reusing connection to ingester");

            Ok(connection.clone())
        } else {
            debug!(%ingester_address, "Connecting to ingester");

            let connection = connection::Builder::new()
                .build(ingester_address)
                .await
                .context(ConnectingSnafu { ingester_address })?;

            // sanity check w/ a handshake
            let mut client =
                LowLevelFlightClient::<proto::IngesterQueryRequest>::new(connection.clone());

            // make contact with the ingester
            client
                .handshake()
                .await
                .context(HandshakeSnafu { ingester_address })?;

            *maybe_connection = Some(connection.clone());
            Ok(connection)
        }
    }
}
