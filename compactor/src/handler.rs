//! Compactor handler

use async_trait::async_trait;
use backoff::{Backoff, BackoffConfig};
use data_types::SequencerId;
use futures::{
    future::{BoxFuture, Shared},
    FutureExt, TryFutureExt,
};
use iox_catalog::interface::Catalog;
use iox_query::exec::Executor;
use iox_time::TimeProvider;
use observability_deps::tracing::*;
use parquet_file::storage::ParquetStorage;
use std::sync::Arc;
use thiserror::Error;
use tokio::task::{JoinError, JoinHandle};
use tokio_util::sync::CancellationToken;

use crate::compact::Compactor;

#[derive(Debug, Error)]
#[allow(missing_copy_implementations, missing_docs)]
pub enum Error {}

/// The [`CompactorHandler`] does nothing at this point
#[async_trait]
pub trait CompactorHandler: Send + Sync {
    /// Wait until the handler finished  to shutdown.
    ///
    /// Use [`shutdown`](Self::shutdown) to trigger a shutdown.
    async fn join(&self);

    /// Shut down background workers.
    fn shutdown(&self);
}

/// A [`JoinHandle`] that can be cloned
type SharedJoinHandle = Shared<BoxFuture<'static, Result<(), Arc<JoinError>>>>;

/// Convert a [`JoinHandle`] into a [`SharedJoinHandle`].
fn shared_handle(handle: JoinHandle<()>) -> SharedJoinHandle {
    handle.map_err(Arc::new).boxed().shared()
}

/// Implementation of the `CompactorHandler` trait (that currently does nothing)
#[derive(Debug)]
pub struct CompactorHandlerImpl {
    /// Data to compact
    #[allow(dead_code)]
    compactor_data: Arc<Compactor>,

    /// A token that is used to trigger shutdown of the background worker
    shutdown: CancellationToken,

    /// Runner to check for compaction work and kick it off
    runner_handle: SharedJoinHandle,
}

impl CompactorHandlerImpl {
    /// Initialize the Compactor
    pub fn new(
        sequencers: Vec<SequencerId>,
        catalog: Arc<dyn Catalog>,
        store: ParquetStorage,
        exec: Arc<Executor>,
        time_provider: Arc<dyn TimeProvider>,
        registry: Arc<metric::Registry>,
        config: CompactorConfig,
    ) -> Self {
        let compactor_data = Arc::new(Compactor::new(
            sequencers,
            catalog,
            store,
            exec,
            time_provider,
            BackoffConfig::default(),
            config,
            registry,
        ));

        let shutdown = CancellationToken::new();
        let runner_handle = tokio::task::spawn(run_compactor(
            Arc::clone(&compactor_data),
            shutdown.child_token(),
        ));
        let runner_handle = shared_handle(runner_handle);
        info!("compactor started with config {:?}", config);

        Self {
            compactor_data,
            shutdown,
            runner_handle,
        }
    }
}

/// The configuration options for the compactor.
#[derive(Debug, Clone, Copy)]
pub struct CompactorConfig {
    /// Max number of level-0 files (written by ingester) we want to compact with level-1 each time
    compaction_max_number_level_0_files: i32,

    /// Desired max size of compacted parquet files
    /// It is a target desired value than a guarantee
    compaction_max_desired_file_size_bytes: i64,

    /// Percentage of desired max file size.
    /// If the estimated compacted result is too small, no need to split it.
    /// This percentage is to determine how small it is:
    ///    < compaction_percentage_max_file_size * compaction_max_desired_file_size_bytes:
    /// This value must be between (0, 100)
    compaction_percentage_max_file_size: i16,

    /// Split file percentage
    /// If the estimated compacted result is neither too small nor too large, it will be split
    /// into 2 files determined by this percentage.
    ///    . Too small means: < compaction_percentage_max_file_size * compaction_max_desired_file_size_bytes
    ///    . Too large means: > compaction_max_desired_file_size_bytes
    ///    . Any size in the middle will be considered neither too small nor too large
    /// This value must be between (0, 100)
    compaction_split_percentage: i16,

    /// The compactor will limit the number of simultaneous compaction jobs based on the
    /// size of the input files to be compacted.  This number should be less than 1/10th
    /// of the available memory to ensure compactions have
    /// enough space to run.
    max_concurrent_compaction_size_bytes: i64,

    /// Max number of partitions per sequencer we want to compact per cycle
    compaction_max_number_partitions_per_sequencer: i32,

    /// Min number of recent writes a partition needs to be considered for compacting
    compaction_min_number_recent_writes_per_partition: i32,
}

impl CompactorConfig {
    /// Initialize a valid config
    pub fn new(
        compaction_max_number_level_0_files: i32,
        compaction_max_desired_file_size_bytes: i64,
        compaction_percentage_max_file_size: i16,
        compaction_split_percentage: i16,
        max_concurrent_compaction_size_bytes: i64,
        compaction_max_number_partitions_per_sequencer: i32,
        compaction_min_number_recent_writes_per_partition: i32,
    ) -> Self {
        assert!(compaction_split_percentage > 0 && compaction_split_percentage <= 100);

        Self {
            compaction_max_number_level_0_files,
            compaction_max_desired_file_size_bytes,
            compaction_percentage_max_file_size,
            compaction_split_percentage,
            max_concurrent_compaction_size_bytes,
            compaction_max_number_partitions_per_sequencer,
            compaction_min_number_recent_writes_per_partition,
        }
    }

    /// Max number of level-0 files we want to compact with level-1 each time
    pub fn compaction_max_number_level_0_files(&self) -> i32 {
        self.compaction_max_number_level_0_files
    }

    /// Desired max file of a compacted file
    pub fn compaction_max_desired_file_size_bytes(&self) -> i64 {
        self.compaction_max_desired_file_size_bytes
    }

    /// Percentage of desired max file size to determine a size is too small
    pub fn compaction_percentage_max_file_size(&self) -> i16 {
        self.compaction_percentage_max_file_size
    }

    /// Percentage of least recent data we want to split to reduce compacting non-overlapped data
    pub fn compaction_split_percentage(&self) -> i16 {
        self.compaction_split_percentage
    }

    /// The compactor will limit the number of simultaneous compaction jobs based on the
    /// size of the input files to be compacted. Currently this only takes into account the
    /// level 0 files, but should later also consider the level 1 files to be compacted. This
    /// number should be less than 1/10th of the available memory to ensure compactions have
    /// enough space to run.
    pub fn max_concurrent_compaction_size_bytes(&self) -> i64 {
        self.max_concurrent_compaction_size_bytes
    }

    /// Max number of partitions per sequencer we want to compact per cycle
    pub fn compaction_max_number_partitions_per_sequencer(&self) -> i32 {
        self.compaction_max_number_partitions_per_sequencer
    }

    /// Min number of recent writes a partition needs to be considered for compacting
    pub fn compaction_min_number_recent_writes_per_partition(&self) -> i32 {
        self.compaction_min_number_recent_writes_per_partition
    }
}

/// Checks for candidate partitions to compact and spawns tokio tasks to compact as many
/// as the configuration will allow. Once those are done it rechecks the catalog for the
/// next top partitions to compact.
async fn run_compactor(compactor: Arc<Compactor>, shutdown: CancellationToken) {
    while !shutdown.is_cancelled() {
        let candidates = Backoff::new(&compactor.backoff_config)
            .retry_all_errors("partitions_to_compact", || async {
                compactor
                    .partitions_to_compact(
                        compactor
                            .config
                            .compaction_max_number_partitions_per_sequencer(),
                        compactor
                            .config
                            .compaction_min_number_recent_writes_per_partition(),
                    )
                    .await
            })
            .await
            .expect("retry forever");
        let candidates = Backoff::new(&compactor.backoff_config)
            .retry_all_errors("partitions_to_compact", || async {
                compactor.add_info_to_partitions(&candidates).await
            })
            .await
            .expect("retry forever");

        let n_candidates = candidates.len();
        debug!(n_candidates, "found compaction candidates");

        // Serially compact all candidates
        // TODO: we will parallelize this when everything runs smoothly in serial
        for c in candidates {
            let compactor = Arc::clone(&compactor);
            let compact_and_upgrade = compactor
                .groups_to_compact_and_files_to_upgrade(
                    c.candidate.partition_id,
                    &c.namespace.name,
                    &c.table.name,
                )
                .await;

            match compact_and_upgrade {
                Err(e) => {
                    warn!(
                        "groups file to compact and upgrade on partition {} failed with: {:?}",
                        c.candidate.partition_id, e
                    );
                }
                Ok(compact_and_upgrade) => {
                    if compact_and_upgrade.compactable() {
                        let res = compactor
                            .compact_partition(
                                &c.namespace,
                                &c.table,
                                &c.table_schema,
                                c.candidate.partition_id,
                                compact_and_upgrade,
                            )
                            .await;
                        if let Err(e) = res {
                            warn!(
                                "compaction on partition {} failed with: {:?}",
                                c.candidate.partition_id, e
                            );
                        }
                        debug!(candidate=?c, "compaction complete");
                    } else {
                        // All candidates should be compactable (have files to compact and/or
                        // upgrade).
                        // Reaching here means we do not choose the right candidates and
                        // it would be a waste of time to repeat this cycle
                        warn!(
                            "The candidate partition {} has no files to be either compacted or \
                                upgraded",
                            c.candidate.partition_id
                        );
                    }
                }
            }
        }
    }
}

#[async_trait]
impl CompactorHandler for CompactorHandlerImpl {
    async fn join(&self) {
        self.runner_handle
            .clone()
            .await
            .expect("compactor task failed");
    }

    fn shutdown(&self) {
        self.shutdown.cancel();
    }
}

impl Drop for CompactorHandlerImpl {
    fn drop(&mut self) {
        if !self.shutdown.is_cancelled() {
            warn!("CompactorHandlerImpl dropped without calling shutdown()");
            self.shutdown.cancel();
        }
    }
}
