use std::sync::atomic::{AtomicUsize, Ordering};

use goose::prelude::*;
use perf_support::SeedManifest;

use crate::cli::{Cli, LoadProfile};
use crate::context;
use crate::http::{ensure_success_response, post_empty_named, post_json_named};
use crate::jobs::{JobSession, enqueue_job, poll_until_terminal};

pub(crate) struct LoadContext {
    pub(crate) cli: Cli,
    pub(crate) manifest: SeedManifest,
    pub(crate) profile: LoadProfile,
    pub(crate) terminal_attempts: AtomicUsize,
    pub(crate) terminal_within_sla: AtomicUsize,
    pub(crate) search_counter: AtomicUsize,
    pub(crate) topic_counter: AtomicUsize,
}

pub(crate) async fn search_request(user: &mut GooseUser) -> TransactionResult {
    let ctx = context();
    let index =
        ctx.search_counter.fetch_add(1, Ordering::Relaxed) % ctx.manifest.search_queries.len();
    let payload = serde_json::json!({
        "query": ctx.manifest.search_queries[index],
        "limit": 10,
    });
    let response = post_json_named(user, "/search", "read_search", &payload).await?;
    ensure_success_response(response)?;
    Ok(())
}

pub(crate) async fn search_filtered_request(user: &mut GooseUser) -> TransactionResult {
    let ctx = context();
    let index =
        ctx.search_counter.fetch_add(1, Ordering::Relaxed) % ctx.manifest.search_queries.len();
    let payload = serde_json::json!({
        "query": ctx.manifest.search_queries[index],
        "limit": 10,
        "filters": {
            "deck_names": [ctx.manifest.duplicate_deck],
            "tags": [ctx.manifest.search_queries[index]],
            "min_reps": 8,
        }
    });
    let response = post_json_named(user, "/search", "read_search_filtered", &payload).await?;
    ensure_success_response(response)?;
    Ok(())
}

pub(crate) async fn search_rerank_request(user: &mut GooseUser) -> TransactionResult {
    let ctx = context();
    let index =
        ctx.search_counter.fetch_add(1, Ordering::Relaxed) % ctx.manifest.search_queries.len();
    let payload = serde_json::json!({
        "query": ctx.manifest.search_queries[index],
        "limit": 10,
        "rerank_override": true,
        "rerank_top_n_override": 5,
    });
    let response = post_json_named(user, "/search", "read_search_rerank", &payload).await?;
    ensure_success_response(response)?;
    Ok(())
}

pub(crate) async fn topics_request(user: &mut GooseUser) -> TransactionResult {
    let ctx = context();
    let index = ctx.topic_counter.fetch_add(1, Ordering::Relaxed) % ctx.manifest.root_topics.len();
    let path = format!("/topics?root_path={}", ctx.manifest.root_topics[index]);
    let response = user.get_named(&path, "read_topics").await?;
    ensure_success_response(response)?;
    Ok(())
}

pub(crate) async fn topic_coverage_request(user: &mut GooseUser) -> TransactionResult {
    let ctx = context();
    let index =
        ctx.topic_counter.fetch_add(1, Ordering::Relaxed) % ctx.manifest.branch_topics.len();
    let path = format!(
        "/topic-coverage?topic_path={}&include_subtree=true",
        ctx.manifest.branch_topics[index]
    );
    let response = user.get_named(&path, "read_topic_coverage").await?;
    ensure_success_response(response)?;
    Ok(())
}

pub(crate) async fn topic_gaps_request(user: &mut GooseUser) -> TransactionResult {
    let ctx = context();
    let path = format!(
        "/topic-gaps?topic_path={}&min_coverage=4",
        ctx.manifest.root_topics[0]
    );
    let response = user.get_named(&path, "read_topic_gaps").await?;
    ensure_success_response(response)?;
    Ok(())
}

pub(crate) async fn topic_weak_notes_request(user: &mut GooseUser) -> TransactionResult {
    let ctx = context();
    let path = format!(
        "/topic-weak-notes?topic_path={}&max_results=20",
        ctx.manifest.root_topics[1]
    );
    let response = user.get_named(&path, "read_topic_weak_notes").await?;
    ensure_success_response(response)?;
    Ok(())
}

pub(crate) async fn duplicates_request(user: &mut GooseUser) -> TransactionResult {
    let path = build_duplicates_path(&context().manifest);
    let response = user.get_named(&path, "read_duplicates").await?;
    ensure_success_response(response)?;
    Ok(())
}

pub(crate) async fn prime_job(user: &mut GooseUser) -> TransactionResult {
    user.set_session_data(JobSession { last_job_id: None });
    let accepted = enqueue_job(
        user,
        "/jobs/sync",
        "job_sync_enqueue",
        &serde_json::json!({
            "source": context().manifest.sync_source,
            "run_migrations": true,
            "index": true,
            "reindex_mode": "incremental",
        }),
    )
    .await?;
    user.get_session_data_mut::<JobSession>()
        .expect("job session")
        .last_job_id = Some(accepted.job_id);
    Ok(())
}

pub(crate) async fn enqueue_sync_job(user: &mut GooseUser) -> TransactionResult {
    let accepted = enqueue_job(
        user,
        "/jobs/sync",
        "job_sync_enqueue",
        &serde_json::json!({
            "source": context().manifest.sync_source,
            "run_migrations": true,
            "index": true,
            "reindex_mode": "incremental",
        }),
    )
    .await?;
    user.get_session_data_mut::<JobSession>()
        .expect("job session")
        .last_job_id = Some(accepted.job_id);
    Ok(())
}

pub(crate) async fn enqueue_index_job(user: &mut GooseUser) -> TransactionResult {
    let accepted = enqueue_job(
        user,
        "/jobs/index",
        "job_index_enqueue",
        &serde_json::json!({
            "reindex_mode": "incremental",
        }),
    )
    .await?;
    user.get_session_data_mut::<JobSession>()
        .expect("job session")
        .last_job_id = Some(accepted.job_id);
    Ok(())
}

pub(crate) async fn job_status_request(user: &mut GooseUser) -> TransactionResult {
    let job_id = if let Some(session) = user.get_session_data::<JobSession>() {
        session.last_job_id.clone()
    } else {
        None
    };
    let job_id = if let Some(job_id) = job_id {
        job_id
    } else {
        let accepted = enqueue_job(
            user,
            "/jobs/sync",
            "job_sync_enqueue",
            &serde_json::json!({
                "source": context().manifest.sync_source,
                "run_migrations": true,
                "index": true,
                "reindex_mode": "incremental",
            }),
        )
        .await?;
        user.get_session_data_mut::<JobSession>()
            .expect("job session")
            .last_job_id = Some(accepted.job_id.clone());
        accepted.job_id
    };

    let terminal = poll_until_terminal(user, &job_id).await?;
    record_terminal_observation(terminal);

    user.get_session_data_mut::<JobSession>()
        .expect("job session")
        .last_job_id = None;
    Ok(())
}

pub(crate) async fn job_cancel_request(user: &mut GooseUser) -> TransactionResult {
    let accepted = enqueue_job(
        user,
        "/jobs/sync",
        "job_sync_enqueue",
        &serde_json::json!({
            "source": context().manifest.sync_source,
            "run_migrations": true,
            "index": true,
            "reindex_mode": "incremental",
        }),
    )
    .await?;
    let path = format!("/jobs/{}/cancel", accepted.job_id);
    let response = post_empty_named(user, &path, "job_cancel").await?;
    ensure_success_response(response)?;
    let terminal = poll_until_terminal(user, &accepted.job_id).await?;
    record_terminal_observation(terminal);
    user.get_session_data_mut::<JobSession>()
        .expect("job session")
        .last_job_id = None;
    Ok(())
}

fn record_terminal_observation(terminal: bool) {
    let ctx = context();
    ctx.terminal_attempts.fetch_add(1, Ordering::Relaxed);
    if terminal {
        ctx.terminal_within_sla.fetch_add(1, Ordering::Relaxed);
    }
}

pub(crate) fn build_duplicates_path(manifest: &SeedManifest) -> String {
    let mut serializer = url::form_urlencoded::Serializer::new(String::new());
    serializer.append_pair("threshold", "0.95");
    serializer.append_pair("max_clusters", "20");
    serializer.append_pair("deck_filter[]", &manifest.duplicate_deck);
    serializer.append_pair("tag_filter[]", &manifest.duplicate_tag);
    format!("/duplicates?{}", serializer.finish())
}
