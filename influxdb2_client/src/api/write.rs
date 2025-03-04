//! Write API

use crate::models::WriteDataPoint;
use crate::{Client, HttpSnafu, RequestError, ReqwestProcessingSnafu};
use futures::{Stream, StreamExt};
use reqwest::{Body, Method};
use snafu::ResultExt;

impl Client {
    /// Write line protocol data to the specified organization and bucket.
    pub async fn write_line_protocol(
        &self,
        org: &str,
        bucket: &str,
        body: impl Into<Body> + Send,
    ) -> Result<(), RequestError> {
        let body = body.into();
        let write_url = format!("{}/api/v2/write", self.url);

        let response = self
            .request(Method::POST, &write_url)
            .query(&[("bucket", bucket), ("org", org)])
            .header("Content-Encoding", "gzip")
            .body(body)
            .send()
            .await
            .context(ReqwestProcessingSnafu)?;

        if !response.status().is_success() {
            let status = response.status();
            let text = response.text().await.context(ReqwestProcessingSnafu)?;
            HttpSnafu { status, text }.fail()?;
        }

        Ok(())
    }

    /// Write a `Stream` of `DataPoint`s to the specified organization and
    /// bucket.
    pub async fn write(
        &self,
        org: &str,
        bucket: &str,
        body: impl Stream<Item = impl WriteDataPoint> + Send + Sync + 'static,
    ) -> Result<(), RequestError> {
        /*
        let mut buffer = bytes::BytesMut::new();

        let body = body.map(move |point| {
            let mut w = (&mut buffer).writer();
            point.write_data_point_to(&mut w)?;
            w.flush()?;
            Ok::<_, io::Error>(buffer.split().freeze())
        });

        let body = Body::wrap_stream(body);
        */
        let mut encoder = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());

        body.for_each(|point| {
            point.write_data_point_to(&mut encoder).unwrap();
            futures::future::ready(())
        }).await;

        let body = Body::from(encoder.finish().unwrap());

        self.write_line_protocol(org, bucket, body).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::DataPoint;
    use futures::stream;
    use mockito::mock;

    #[tokio::test]
    async fn writing_points() {
        let org = "some-org";
        let bucket = "some-bucket";
        let token = "some-token";

        let mock_server = mock(
            "POST",
            format!("/api/v2/write?bucket={}&org={}", bucket, org).as_str(),
        )
        .match_header("Authorization", format!("Token {}", token).as_str())
        // .match_header("Content-Encoding", "gzip")
        .match_body(
            "\
cpu,host=server01 usage=0.5
cpu,host=server01,region=us-west usage=0.87
",
        )
        .create();

        let client = Client::new(&mockito::server_url(), token);

        let points = vec![
            DataPoint::builder("cpu")
                .tag("host", "server01")
                .field("usage", 0.5)
                .build()
                .unwrap(),
            DataPoint::builder("cpu")
                .tag("host", "server01")
                .tag("region", "us-west")
                .field("usage", 0.87)
                .build()
                .unwrap(),
        ];

        // If the requests made are incorrect, Mockito returns status 501 and `write`
        // will return an error, which causes the test to fail here instead of
        // when we assert on mock_server. The error messages that Mockito
        // provides are much clearer for explaining why a test failed than just
        // that the server returned 501, so don't use `?` here.
        let _result = client.write(org, bucket, stream::iter(points)).await;

        mock_server.assert();
    }
}
