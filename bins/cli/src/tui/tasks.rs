use std::sync::Arc;

use common::ReindexMode;
use surface_runtime::{SurfaceProgressEvent, SurfaceProgressSink};
use tokio::sync::mpsc;

use crate::usecases::{
    self, CoverageRequest, DuplicatesRequest, GapsRequest, GenerateRequest, IndexRequest,
    ObsidianScanRequest, SyncRequest, TagAuditRequest, TopicsTreeRequest, ValidateRequest,
    WeakNotesRequest,
};

use super::{AppEvent, AppState, TaskResult, TopicsTab, WorkflowTab, split_csv};

pub(crate) fn run_topics_task(app: &mut AppState, tx: &mpsc::UnboundedSender<AppEvent>) {
    let Some(handles) = app.runtime_handles() else {
        return;
    };

    match app.topics.tab {
        TopicsTab::Tree => {
            let request = TopicsTreeRequest {
                root_path: (!app.topics.tree_root.trim().is_empty())
                    .then(|| app.topics.tree_root.trim().to_string()),
            };
            app.busy_label = Some("topics tree".to_string());
            spawn_task(tx.clone(), "topics tree", move || async move {
                usecases::topics_tree(handles, request)
                    .await
                    .map(TaskResult::TopicsTree)
                    .map_err(|error| error.to_string())
            });
        }
        TopicsTab::Coverage => {
            let request = CoverageRequest {
                topic: app.topics.coverage_topic.clone(),
                include_subtree: app.topics.coverage_include_subtree,
            };
            app.busy_label = Some("coverage".to_string());
            spawn_task(tx.clone(), "coverage", move || async move {
                usecases::coverage(handles, request)
                    .await
                    .map(TaskResult::Coverage)
                    .map_err(|error| error.to_string())
            });
        }
        TopicsTab::Gaps => {
            let request = GapsRequest {
                topic: app.topics.gaps_topic.clone(),
                min_coverage: app.topics.gaps_min_coverage.parse::<i64>().unwrap_or(1),
            };
            app.busy_label = Some("gaps".to_string());
            spawn_task(tx.clone(), "gaps", move || async move {
                usecases::gaps(handles, request)
                    .await
                    .map(TaskResult::Gaps)
                    .map_err(|error| error.to_string())
            });
        }
        TopicsTab::WeakNotes => {
            let request = WeakNotesRequest {
                topic: app.topics.weak_topic.clone(),
                limit: app.topics.weak_limit.parse::<i64>().unwrap_or(20),
            };
            app.busy_label = Some("weak notes".to_string());
            spawn_task(tx.clone(), "weak notes", move || async move {
                usecases::weak_notes(handles, request)
                    .await
                    .map(TaskResult::WeakNotes)
                    .map_err(|error| error.to_string())
            });
        }
        TopicsTab::Duplicates => {
            let request = DuplicatesRequest {
                threshold: app
                    .topics
                    .duplicates_threshold
                    .parse::<f64>()
                    .unwrap_or(0.92),
                max: app.topics.duplicates_max.parse::<usize>().unwrap_or(50),
                deck_names: split_csv(&app.topics.duplicates_decks),
                tags: split_csv(&app.topics.duplicates_tags),
            };
            app.busy_label = Some("duplicates".to_string());
            spawn_task(tx.clone(), "duplicates", move || async move {
                usecases::duplicates(handles, request)
                    .await
                    .map(|(clusters, stats)| TaskResult::Duplicates(clusters, stats))
                    .map_err(|error| error.to_string())
            });
        }
    }
}

pub(crate) fn run_workflow_task(app: &mut AppState, tx: &mpsc::UnboundedSender<AppEvent>) {
    let Some(handles) = app.runtime_handles() else {
        return;
    };

    match app.workflows.tab {
        WorkflowTab::Sync => {
            let request = SyncRequest {
                source: app.workflows.sync_source.clone().into(),
                run_migrations: app.workflows.sync_run_migrations,
                run_index: app.workflows.sync_run_index,
                reindex_mode: if app.workflows.sync_force_reindex {
                    ReindexMode::Force
                } else {
                    ReindexMode::Incremental
                },
            };
            let progress = progress_sink(tx.clone());
            app.busy_label = Some("sync".to_string());
            spawn_task(tx.clone(), "sync", move || async move {
                usecases::sync(handles, request, Some(progress))
                    .await
                    .map(TaskResult::Sync)
                    .map_err(|error| error.to_string())
            });
        }
        WorkflowTab::Index => {
            let request = IndexRequest {
                reindex_mode: if app.workflows.index_force_reindex {
                    ReindexMode::Force
                } else {
                    ReindexMode::Incremental
                },
            };
            let progress = progress_sink(tx.clone());
            app.busy_label = Some("index".to_string());
            spawn_task(tx.clone(), "index", move || async move {
                usecases::index(handles, request, Some(progress))
                    .await
                    .map(TaskResult::Index)
                    .map_err(|error| error.to_string())
            });
        }
        WorkflowTab::Generate => {
            let request = GenerateRequest {
                file: app.workflows.generate_file.clone().into(),
            };
            let service = Arc::clone(&handles.generate_preview);
            app.busy_label = Some("generate preview".to_string());
            spawn_task(tx.clone(), "generate preview", move || async move {
                usecases::generate_preview(&service, &request)
                    .map(TaskResult::Generate)
                    .map_err(|error| error.to_string())
            });
        }
        WorkflowTab::Validate => {
            let request = ValidateRequest {
                file: app.workflows.validate_file.clone().into(),
                include_quality: if app.workflows.validate_quality {
                    surface_runtime::QualityCheck::Include
                } else {
                    surface_runtime::QualityCheck::Skip
                },
            };
            let service = Arc::clone(&handles.validation);
            app.busy_label = Some("validate".to_string());
            spawn_task(tx.clone(), "validate", move || async move {
                usecases::validate(&service, &request)
                    .map(TaskResult::Validate)
                    .map_err(|error| error.to_string())
            });
        }
        WorkflowTab::Obsidian => {
            let request = ObsidianScanRequest {
                vault: app.workflows.obsidian_vault.clone().into(),
                source_dirs: split_csv(&app.workflows.obsidian_source_dirs),
                execution_mode: app.workflows.obsidian_dry_run,
            };
            let service = Arc::clone(&handles.obsidian_scan);
            let progress_tx = tx.clone();
            app.busy_label = Some("obsidian scan".to_string());
            spawn_task(tx.clone(), "obsidian scan", move || async move {
                usecases::obsidian_scan(
                    &service,
                    &request,
                    Some(progress_sink(progress_tx.clone())),
                )
                .map(TaskResult::Obsidian)
                .map_err(|error| error.to_string())
            });
        }
        WorkflowTab::TagAudit => {
            let request = TagAuditRequest {
                file: app.workflows.tag_audit_file.clone().into(),
                apply_fixes: app.workflows.tag_audit_fix,
            };
            let service = Arc::clone(&handles.tag_audit);
            app.busy_label = Some("tag audit".to_string());
            spawn_task(tx.clone(), "tag audit", move || async move {
                usecases::tag_audit(&service, &request)
                    .map(TaskResult::TagAudit)
                    .map_err(|error| error.to_string())
            });
        }
    }
}

fn progress_sink(tx: mpsc::UnboundedSender<AppEvent>) -> SurfaceProgressSink {
    Arc::new(move |event: SurfaceProgressEvent| {
        let _ = tx.send(AppEvent::Progress(event));
    })
}

pub(crate) fn spawn_task<F, Fut>(tx: mpsc::UnboundedSender<AppEvent>, label: &'static str, task: F)
where
    F: FnOnce() -> Fut + Send + 'static,
    Fut: std::future::Future<Output = Result<TaskResult, String>> + 'static,
{
    std::thread::spawn(move || {
        let result = match tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
        {
            Ok(runtime) => runtime.block_on(task()).map_err(|error| error.to_string()),
            Err(error) => Err(error.to_string()),
        };
        let _ = tx.send(AppEvent::TaskFinished {
            label: label.to_string(),
            result: Box::new(result),
        });
    });
}
