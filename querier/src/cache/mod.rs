//! Caches used by the querier.
use backoff::BackoffConfig;
use cache_system::backend::lru::ResourcePool;
use iox_catalog::interface::Catalog;
use iox_time::TimeProvider;
use std::sync::Arc;

use self::{
    namespace::NamespaceCache, parquet_file::ParquetFileCache, partition::PartitionCache,
    processed_tombstones::ProcessedTombstonesCache, ram::RamSize, read_buffer::ReadBufferCache,
    table::TableCache, tombstones::TombstoneCache,
};

pub mod namespace;
pub mod parquet_file;
pub mod partition;
pub mod processed_tombstones;
mod ram;
pub mod read_buffer;
pub mod table;
pub mod tombstones;

#[cfg(test)]
mod test_util;

/// Caches request to the [`Catalog`].
#[derive(Debug)]
pub struct CatalogCache {
    /// Catalog.
    catalog: Arc<dyn Catalog>,

    /// Partition cache.
    partition_cache: PartitionCache,

    /// Table cache.
    table_cache: TableCache,

    /// Namespace cache.
    namespace_cache: NamespaceCache,

    /// Processed tombstone cache.
    processed_tombstones_cache: ProcessedTombstonesCache,

    /// Parquet file cache
    parquet_file_cache: ParquetFileCache,

    /// tombstone cache
    tombstone_cache: TombstoneCache,

    /// Read buffer chunk cache
    read_buffer_cache: ReadBufferCache,

    /// Metric registry
    metric_registry: Arc<metric::Registry>,

    /// Time provider.
    time_provider: Arc<dyn TimeProvider>,
}

impl CatalogCache {
    /// Create empty cache.
    pub fn new(
        catalog: Arc<dyn Catalog>,
        time_provider: Arc<dyn TimeProvider>,
        metric_registry: Arc<metric::Registry>,
        ram_pool_bytes: usize,
    ) -> Self {
        Self::new_internal(
            catalog,
            time_provider,
            metric_registry,
            ram_pool_bytes,
            false,
        )
    }

    /// Create empty cache for testing.
    pub fn new_testing(
        catalog: Arc<dyn Catalog>,
        time_provider: Arc<dyn TimeProvider>,
        metric_registry: Arc<metric::Registry>,
        ram_pool_bytes: usize,
    ) -> Self {
        Self::new_internal(
            catalog,
            time_provider,
            metric_registry,
            ram_pool_bytes,
            true,
        )
    }

    fn new_internal(
        catalog: Arc<dyn Catalog>,
        time_provider: Arc<dyn TimeProvider>,
        metric_registry: Arc<metric::Registry>,
        ram_pool_bytes: usize,
        testing: bool,
    ) -> Self {
        let backoff_config = BackoffConfig::default();
        let ram_pool = Arc::new(ResourcePool::new(
            "ram",
            RamSize(ram_pool_bytes),
            Arc::clone(&time_provider),
            Arc::clone(&metric_registry),
        ));

        let partition_cache = PartitionCache::new(
            Arc::clone(&catalog),
            backoff_config.clone(),
            Arc::clone(&time_provider),
            &metric_registry,
            Arc::clone(&ram_pool),
            testing,
        );
        let table_cache = TableCache::new(
            Arc::clone(&catalog),
            backoff_config.clone(),
            Arc::clone(&time_provider),
            &metric_registry,
            Arc::clone(&ram_pool),
            testing,
        );
        let namespace_cache = NamespaceCache::new(
            Arc::clone(&catalog),
            backoff_config.clone(),
            Arc::clone(&time_provider),
            &metric_registry,
            Arc::clone(&ram_pool),
            testing,
        );
        let processed_tombstones_cache = ProcessedTombstonesCache::new(
            Arc::clone(&catalog),
            backoff_config.clone(),
            Arc::clone(&time_provider),
            &metric_registry,
            Arc::clone(&ram_pool),
            testing,
        );
        let parquet_file_cache = ParquetFileCache::new(
            Arc::clone(&catalog),
            backoff_config.clone(),
            Arc::clone(&time_provider),
            &metric_registry,
            Arc::clone(&ram_pool),
            testing,
        );
        let tombstone_cache = TombstoneCache::new(
            Arc::clone(&catalog),
            backoff_config.clone(),
            Arc::clone(&time_provider),
            &metric_registry,
            Arc::clone(&ram_pool),
            testing,
        );
        let read_buffer_cache = ReadBufferCache::new(
            backoff_config,
            Arc::clone(&time_provider),
            Arc::clone(&metric_registry),
            Arc::clone(&ram_pool),
            testing,
        );

        Self {
            catalog,
            partition_cache,
            table_cache,
            namespace_cache,
            processed_tombstones_cache,
            parquet_file_cache,
            tombstone_cache,
            read_buffer_cache,
            metric_registry,
            time_provider,
        }
    }

    /// Get underlying catalog
    pub(crate) fn catalog(&self) -> Arc<dyn Catalog> {
        Arc::clone(&self.catalog)
    }

    /// Get underlying metric registry.
    pub(crate) fn metric_registry(&self) -> Arc<metric::Registry> {
        Arc::clone(&self.metric_registry)
    }

    /// Get underlying time provider
    pub(crate) fn time_provider(&self) -> Arc<dyn TimeProvider> {
        Arc::clone(&self.time_provider)
    }

    /// Namespace cache
    pub(crate) fn namespace(&self) -> &NamespaceCache {
        &self.namespace_cache
    }

    /// Table cache
    pub(crate) fn table(&self) -> &TableCache {
        &self.table_cache
    }

    /// Partition cache
    pub(crate) fn partition(&self) -> &PartitionCache {
        &self.partition_cache
    }

    /// Processed tombstone cache.
    pub(crate) fn processed_tombstones(&self) -> &ProcessedTombstonesCache {
        &self.processed_tombstones_cache
    }

    /// Parquet file cache.
    pub(crate) fn parquet_file(&self) -> &ParquetFileCache {
        &self.parquet_file_cache
    }

    /// Tombstone cache.
    pub(crate) fn tombstone(&self) -> &TombstoneCache {
        &self.tombstone_cache
    }

    /// Read buffer chunk cache.
    pub(crate) fn read_buffer(&self) -> &ReadBufferCache {
        &self.read_buffer_cache
    }
}
