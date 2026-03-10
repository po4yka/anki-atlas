use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::path::PathBuf;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use anyhow::{Context, Result};
use goose::config::{GooseConfiguration, GooseDefault};
use goose::goose::GooseResponse;
use goose::metrics::{GooseMetrics, GooseRequestMetricAggregate};
use goose::prelude::*;
use gumdrop::Options;
use perf_support::{DatasetProfile, SeedManifest, manifest_json, profile_manifest, reset_and_seed};
use serde::{Deserialize, Serialize};

static CONTEXT: OnceLock<LoadContext> = OnceLock::new();

#[derive(Debug, Clone)]
struct Cli {
    profile: DatasetProfile,
    scenario: ScenarioMode,
    base_url: String,
    report_json: PathBuf,
    prepare_only: bool,
}

#[derive(Debug, Clone, Copy)]
enum ScenarioMode {
    Read,
    Jobs,
    Full,
}

impl ScenarioMode {
    fn as_str(self) -> &'static str {
        match self {
            Self::Read => "read",
            Self::Jobs => "jobs",
            Self::Full => "full",
        }
    }

    fn includes_read(self) -> bool {
        matches!(self, Self::Read | Self::Full)
    }

    fn includes_jobs(self) -> bool {
        matches!(self, Self::Jobs | Self::Full)
    }
}

#[derive(Debug, Clone, Copy)]
struct LoadProfile {
    total_users: usize,
    hatch_rate: &'static str,
    run_time_secs: usize,
    read_p95_ms: usize,
    job_p95_ms: usize,
    terminal_ratio_min: f64,
    terminal_sla_secs: usize,
    report_only: bool,
}

impl LoadProfile {
    fn for_profile(profile: DatasetProfile) -> Self {
        match profile {
            DatasetProfile::Pr => Self {
                total_users: 18,
                hatch_rate: "6",
                run_time_secs: 120,
                read_p95_ms: 1_500,
                job_p95_ms: 750,
                terminal_ratio_min: 0.95,
                terminal_sla_secs: 5,
                report_only: false,
            },
            DatasetProfile::Nightly => Self {
                total_users: 70,
                hatch_rate: "20",
                run_time_secs: 600,
                read_p95_ms: 1_500,
                job_p95_ms: 750,
                terminal_ratio_min: 0.95,
                terminal_sla_secs: 5,
                report_only: true,
            },
        }
    }
}

struct LoadContext {
    cli: Cli,
    manifest: SeedManifest,
    profile: LoadProfile,
    terminal_attempts: AtomicUsize,
    terminal_within_sla: AtomicUsize,
    search_counter: AtomicUsize,
    topic_counter: AtomicUsize,
}

#[derive(Clone)]
struct JobSession {
    last_job_id: Option<String>,
}

#[derive(Debug, Deserialize)]
struct JobAcceptedResponse {
    job_id: String,
}

#[derive(Debug, Deserialize)]
struct JobStatusResponse {
    status: String,
}

#[derive(Debug, Serialize)]
struct PerfReport {
    profile: String,
    scenario: String,
    base_url: String,
    report_only: bool,
    duration_secs: usize,
    total_users: usize,
    requests_seen: usize,
    read: RequestGroupSummary,
    jobs: RequestGroupSummary,
    worker_terminal_ratio: f64,
    worker_terminal_attempts: usize,
    thresholds: ThresholdReport,
}

#[derive(Debug, Serialize, Default)]
struct RequestGroupSummary {
    requests: usize,
    successes: usize,
    failures: usize,
    error_rate: f64,
    p95_ms: usize,
}

#[derive(Debug, Serialize)]
struct ThresholdReport {
    read_evaluated: bool,
    read_error_rate_ok: bool,
    read_p95_ok: bool,
    job_evaluated: bool,
    job_error_rate_ok: bool,
    job_p95_ok: bool,
    worker_terminal_evaluated: bool,
    worker_terminal_ok: bool,
}

fn context() -> &'static LoadContext {
    CONTEXT.get().expect("perf harness context not initialized")
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = parse_cli()?;
    let manifest = profile_manifest(cli.profile);
    fs::create_dir_all(
        cli.report_json
            .parent()
            .unwrap_or_else(|| std::path::Path::new("target/perf")),
    )
    .context("create perf report directory")?;

    if cli.prepare_only {
        let settings =
            common::config::Settings::load().context("load settings for perf seeding")?;
        let seeded_manifest = reset_and_seed(&settings, cli.profile)
            .await
            .context("seed runtime for performance tests")?;
        let manifest_path = manifest_path_for(cli.profile);
        fs::write(&manifest_path, manifest_json(&seeded_manifest)?)
            .with_context(|| format!("write seed manifest to {}", manifest_path.display()))?;
        println!("prepared {}", manifest_path.display());
        return Ok(());
    }

    let load_profile = LoadProfile::for_profile(cli.profile);
    CONTEXT
        .set(LoadContext {
            cli: cli.clone(),
            manifest,
            profile: load_profile,
            terminal_attempts: AtomicUsize::new(0),
            terminal_within_sla: AtomicUsize::new(0),
            search_counter: AtomicUsize::new(0),
            topic_counter: AtomicUsize::new(0),
        })
        .map_err(|_| anyhow::anyhow!("perf harness context already initialized"))?;

    let config = GooseConfiguration::parse_args_default(&["perf-harness", "--quiet"])
        .context("build Goose configuration")?;

    let mut attack = GooseAttack::initialize_with_config(config)
        .context("initialize Goose load test")?
        .set_default(GooseDefault::Host, cli.base_url.as_str())?
        .set_default(GooseDefault::Users, load_profile.total_users)?
        .set_default(GooseDefault::RunTime, load_profile.run_time_secs)?
        .set_default(GooseDefault::HatchRate, load_profile.hatch_rate)?
        .set_default(GooseDefault::NoResetMetrics, true)?;

    attack = register_scenarios(attack, cli.scenario)?;

    let metrics = attack.execute().await.context("execute Goose load test")?;
    let report = build_report(&metrics);
    fs::write(
        &cli.report_json,
        serde_json::to_string_pretty(&report).context("serialize perf report")?,
    )
    .with_context(|| format!("write perf report to {}", cli.report_json.display()))?;

    print_report(&report);

    if !load_profile.report_only {
        enforce_thresholds(&report)?;
    }

    Ok(())
}

fn parse_cli() -> Result<Cli> {
    let mut profile = DatasetProfile::Pr;
    let mut scenario = ScenarioMode::Full;
    let mut base_url =
        env::var("ANKIATLAS_PERF_BASE_URL").unwrap_or_else(|_| "http://127.0.0.1:8000".to_string());
    let mut report_json = PathBuf::new();
    let mut prepare_only = false;

    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--profile" => {
                let value = args.next().context("missing value for --profile")?;
                profile = value.parse()?;
            }
            "--scenario" => {
                let value = args.next().context("missing value for --scenario")?;
                scenario = match value.as_str() {
                    "read" => ScenarioMode::Read,
                    "jobs" => ScenarioMode::Jobs,
                    "full" => ScenarioMode::Full,
                    other => anyhow::bail!("unsupported scenario: {other}"),
                };
            }
            "--base-url" => {
                base_url = args.next().context("missing value for --base-url")?;
            }
            "--report-json" => {
                report_json =
                    PathBuf::from(args.next().context("missing value for --report-json")?);
            }
            "--prepare-only" => {
                prepare_only = true;
            }
            "--help" | "-h" => {
                print_usage();
                std::process::exit(0);
            }
            other => anyhow::bail!("unsupported argument: {other}"),
        }
    }

    if report_json.as_os_str().is_empty() {
        report_json = PathBuf::from(format!(
            "target/perf/{}-{}.json",
            profile.as_str(),
            scenario.as_str()
        ));
    }

    Ok(Cli {
        profile,
        scenario,
        base_url,
        report_json,
        prepare_only,
    })
}

fn print_usage() {
    println!(
        "perf-harness --profile <pr|nightly> --scenario <read|jobs|full> [--base-url URL] [--report-json FILE] [--prepare-only]"
    );
}

fn manifest_path_for(profile: DatasetProfile) -> PathBuf {
    PathBuf::from(format!(
        "target/perf/seed-manifest-{}.json",
        profile.as_str()
    ))
}

fn register_scenarios(
    attack: Box<GooseAttack>,
    scenario_mode: ScenarioMode,
) -> std::result::Result<Box<GooseAttack>, GooseError> {
    let read_scenario = scenario!("read")
        .set_weight(2)?
        .set_wait_time(Duration::from_millis(25), Duration::from_millis(125))?
        .register_transaction(transaction!(search_request).set_weight(25)?)
        .register_transaction(transaction!(search_filtered_request).set_weight(10)?)
        .register_transaction(transaction!(search_rerank_request).set_weight(5)?)
        .register_transaction(transaction!(topics_request).set_weight(15)?)
        .register_transaction(transaction!(topic_coverage_request).set_weight(15)?)
        .register_transaction(transaction!(topic_gaps_request).set_weight(10)?)
        .register_transaction(transaction!(topic_weak_notes_request).set_weight(10)?)
        .register_transaction(transaction!(duplicates_request).set_weight(10)?);

    let jobs_scenario = scenario!("jobs")
        .set_weight(1)?
        .set_wait_time(Duration::from_millis(20), Duration::from_millis(80))?
        .register_transaction(transaction!(prime_job).set_on_start())
        .register_transaction(transaction!(enqueue_sync_job).set_weight(30)?)
        .register_transaction(transaction!(enqueue_index_job).set_weight(20)?)
        .register_transaction(transaction!(job_status_request).set_weight(30)?)
        .register_transaction(transaction!(job_cancel_request).set_weight(20)?);

    match scenario_mode {
        ScenarioMode::Read => Ok(Box::new(attack.register_scenario(read_scenario))),
        ScenarioMode::Jobs => Ok(Box::new(attack.register_scenario(jobs_scenario))),
        ScenarioMode::Full => Ok(Box::new(
            attack
                .register_scenario(read_scenario)
                .register_scenario(jobs_scenario),
        )),
    }
}

async fn search_request(user: &mut GooseUser) -> TransactionResult {
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

async fn search_filtered_request(user: &mut GooseUser) -> TransactionResult {
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

async fn search_rerank_request(user: &mut GooseUser) -> TransactionResult {
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

async fn topics_request(user: &mut GooseUser) -> TransactionResult {
    let ctx = context();
    let index = ctx.topic_counter.fetch_add(1, Ordering::Relaxed) % ctx.manifest.root_topics.len();
    let path = format!("/topics?root_path={}", ctx.manifest.root_topics[index]);
    let response = user.get_named(&path, "read_topics").await?;
    ensure_success_response(response)?;
    Ok(())
}

async fn topic_coverage_request(user: &mut GooseUser) -> TransactionResult {
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

async fn topic_gaps_request(user: &mut GooseUser) -> TransactionResult {
    let ctx = context();
    let path = format!(
        "/topic-gaps?topic_path={}&min_coverage=4",
        ctx.manifest.root_topics[0]
    );
    let response = user.get_named(&path, "read_topic_gaps").await?;
    ensure_success_response(response)?;
    Ok(())
}

async fn topic_weak_notes_request(user: &mut GooseUser) -> TransactionResult {
    let ctx = context();
    let path = format!(
        "/topic-weak-notes?topic_path={}&max_results=20",
        ctx.manifest.root_topics[1]
    );
    let response = user.get_named(&path, "read_topic_weak_notes").await?;
    ensure_success_response(response)?;
    Ok(())
}

async fn duplicates_request(user: &mut GooseUser) -> TransactionResult {
    let path = build_duplicates_path(&context().manifest);
    let response = user.get_named(&path, "read_duplicates").await?;
    ensure_success_response(response)?;
    Ok(())
}

async fn prime_job(user: &mut GooseUser) -> TransactionResult {
    user.set_session_data(JobSession { last_job_id: None });
    let accepted = enqueue_job(
        user,
        "/jobs/sync",
        "job_sync_enqueue",
        &serde_json::json!({
            "source": context().manifest.sync_source,
            "run_migrations": true,
            "index": true,
            "force_reindex": false,
        }),
    )
    .await?;
    user.get_session_data_mut::<JobSession>()
        .expect("job session")
        .last_job_id = Some(accepted.job_id);
    Ok(())
}

async fn enqueue_sync_job(user: &mut GooseUser) -> TransactionResult {
    let accepted = enqueue_job(
        user,
        "/jobs/sync",
        "job_sync_enqueue",
        &serde_json::json!({
            "source": context().manifest.sync_source,
            "run_migrations": true,
            "index": true,
            "force_reindex": false,
        }),
    )
    .await?;
    user.get_session_data_mut::<JobSession>()
        .expect("job session")
        .last_job_id = Some(accepted.job_id);
    Ok(())
}

async fn enqueue_index_job(user: &mut GooseUser) -> TransactionResult {
    let accepted = enqueue_job(
        user,
        "/jobs/index",
        "job_index_enqueue",
        &serde_json::json!({
            "force_reindex": false,
        }),
    )
    .await?;
    user.get_session_data_mut::<JobSession>()
        .expect("job session")
        .last_job_id = Some(accepted.job_id);
    Ok(())
}

async fn job_status_request(user: &mut GooseUser) -> TransactionResult {
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
                "force_reindex": false,
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

async fn job_cancel_request(user: &mut GooseUser) -> TransactionResult {
    let accepted = enqueue_job(
        user,
        "/jobs/sync",
        "job_sync_enqueue",
        &serde_json::json!({
            "source": context().manifest.sync_source,
            "run_migrations": true,
            "index": true,
            "force_reindex": false,
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

fn build_duplicates_path(manifest: &SeedManifest) -> String {
    let mut serializer = url::form_urlencoded::Serializer::new(String::new());
    serializer.append_pair("threshold", "0.95");
    serializer.append_pair("max_clusters", "20");
    serializer.append_pair("deck_filter[]", &manifest.duplicate_deck);
    serializer.append_pair("tag_filter[]", &manifest.duplicate_tag);
    format!("/duplicates?{}", serializer.finish())
}

async fn enqueue_job(
    user: &mut GooseUser,
    path: &str,
    name: &str,
    payload: &serde_json::Value,
) -> std::result::Result<JobAcceptedResponse, Box<TransactionError>> {
    let response = post_json_named(user, path, name, payload).await?;
    let response = ensure_success_response(response)?;
    match response.response {
        Ok(response) => response
            .json::<JobAcceptedResponse>()
            .await
            .map_err(|error| Box::new(error.into()) as Box<TransactionError>),
        Err(error) => Err(Box::new(error.into())),
    }
}

async fn poll_until_terminal(
    user: &mut GooseUser,
    job_id: &str,
) -> std::result::Result<bool, Box<TransactionError>> {
    let deadline = tokio::time::Instant::now()
        + Duration::from_secs(context().profile.terminal_sla_secs as u64);
    let path = format!("/jobs/{job_id}");

    loop {
        let response = user.get_named(&path, "job_status").await?;
        let response = ensure_success_response(response)?;
        let payload = match response.response {
            Ok(response) => response
                .json::<JobStatusResponse>()
                .await
                .map_err(|error| Box::new(error.into()) as Box<TransactionError>)?,
            Err(error) => return Err(Box::new(error.into())),
        };

        if matches!(
            payload.status.as_str(),
            "failed" | "cancelled" | "succeeded"
        ) {
            return Ok(true);
        }
        if tokio::time::Instant::now() >= deadline {
            return Ok(false);
        }
        tokio::time::sleep(Duration::from_millis(200)).await;
    }
}

async fn post_json_named<T: serde::Serialize + ?Sized>(
    user: &mut GooseUser,
    path: &str,
    name: &str,
    payload: &T,
) -> std::result::Result<GooseResponse, Box<TransactionError>> {
    let request_builder = user
        .get_request_builder(&GooseMethod::Post, path)?
        .json(payload);
    let goose_request = GooseRequest::builder()
        .method(GooseMethod::Post)
        .path(path)
        .name(name)
        .set_request_builder(request_builder)
        .build();
    user.request(goose_request).await
}

async fn post_empty_named(
    user: &mut GooseUser,
    path: &str,
    name: &str,
) -> std::result::Result<GooseResponse, Box<TransactionError>> {
    let request_builder = user
        .get_request_builder(&GooseMethod::Post, path)?
        .body(String::new());
    let goose_request = GooseRequest::builder()
        .method(GooseMethod::Post)
        .path(path)
        .name(name)
        .set_request_builder(request_builder)
        .build();
    user.request(goose_request).await
}

fn ensure_success_response(
    response: GooseResponse,
) -> std::result::Result<GooseResponse, Box<TransactionError>> {
    match response.response.as_ref() {
        Ok(inner) if inner.status().is_success() => Ok(response),
        Ok(_inner) => Err(Box::new(TransactionError::RequestFailed {
            raw_request: response.request.clone(),
        })),
        Err(_error) => Err(Box::new(TransactionError::RequestFailed {
            raw_request: response.request.clone(),
        })),
    }
}

fn build_report(metrics: &GooseMetrics) -> PerfReport {
    let ctx = context();
    let read = summarize_requests(
        metrics
            .requests
            .iter()
            .map(|(_, request)| request)
            .filter(|request| request.path.starts_with("read_")),
    );
    let jobs = summarize_requests(
        metrics
            .requests
            .iter()
            .map(|(_, request)| request)
            .filter(|request| request.path.starts_with("job_")),
    );

    let attempts = ctx.terminal_attempts.load(Ordering::Relaxed);
    let within_sla = ctx.terminal_within_sla.load(Ordering::Relaxed);
    let worker_terminal_ratio = if attempts == 0 {
        0.0
    } else {
        within_sla as f64 / attempts as f64
    };

    let thresholds = ThresholdReport {
        read_evaluated: ctx.cli.scenario.includes_read(),
        read_error_rate_ok: !ctx.cli.scenario.includes_read() || read.error_rate <= 0.01,
        read_p95_ok: !ctx.cli.scenario.includes_read() || read.p95_ms <= ctx.profile.read_p95_ms,
        job_evaluated: ctx.cli.scenario.includes_jobs(),
        job_error_rate_ok: !ctx.cli.scenario.includes_jobs() || jobs.error_rate <= 0.01,
        job_p95_ok: !ctx.cli.scenario.includes_jobs() || jobs.p95_ms <= ctx.profile.job_p95_ms,
        worker_terminal_evaluated: ctx.cli.scenario.includes_jobs(),
        worker_terminal_ok: !ctx.cli.scenario.includes_jobs()
            || worker_terminal_ratio >= ctx.profile.terminal_ratio_min,
    };

    PerfReport {
        profile: ctx.cli.profile.as_str().to_string(),
        scenario: ctx.cli.scenario.as_str().to_string(),
        base_url: ctx.cli.base_url.clone(),
        report_only: ctx.profile.report_only,
        duration_secs: metrics.duration,
        total_users: metrics.maximum_users,
        requests_seen: read.requests + jobs.requests,
        read,
        jobs,
        worker_terminal_ratio,
        worker_terminal_attempts: attempts,
        thresholds,
    }
}

fn summarize_requests<'a>(
    requests: impl Iterator<Item = &'a GooseRequestMetricAggregate>,
) -> RequestGroupSummary {
    let mut merged_times: BTreeMap<usize, usize> = BTreeMap::new();
    let mut summary = RequestGroupSummary::default();

    for request in requests {
        summary.requests += request.raw_data.counter;
        summary.successes += request.success_count;
        summary.failures += request.fail_count;

        for (millis, count) in &request.raw_data.times {
            *merged_times.entry(*millis).or_insert(0) += count;
        }
    }

    if summary.requests > 0 {
        summary.error_rate = summary.failures as f64 / summary.requests as f64;
        summary.p95_ms = percentile(&merged_times, 0.95);
    }

    summary
}

fn percentile(histogram: &BTreeMap<usize, usize>, percentile: f64) -> usize {
    if histogram.is_empty() {
        return 0;
    }

    let total = histogram.values().copied().sum::<usize>();
    let target = (total as f64 * percentile).ceil() as usize;
    let mut seen = 0_usize;

    for (millis, count) in histogram {
        seen += count;
        if seen >= target {
            return *millis;
        }
    }

    *histogram.keys().last().unwrap_or(&0)
}

fn print_report(report: &PerfReport) {
    println!(
        "profile={} scenario={} users={} read_p95={}ms read_error_rate={:.3} job_p95={}ms job_error_rate={:.3} terminal_ratio={:.3}",
        report.profile,
        report.scenario,
        report.total_users,
        report.read.p95_ms,
        report.read.error_rate,
        report.jobs.p95_ms,
        report.jobs.error_rate,
        report.worker_terminal_ratio
    );
}

fn enforce_thresholds(report: &PerfReport) -> Result<()> {
    let thresholds = &report.thresholds;
    if thresholds.read_error_rate_ok
        && thresholds.read_p95_ok
        && thresholds.job_error_rate_ok
        && thresholds.job_p95_ok
        && thresholds.worker_terminal_ok
    {
        return Ok(());
    }

    anyhow::bail!(
        "performance smoke thresholds failed: read_evaluated={} read_error_rate_ok={} read_p95_ok={} job_evaluated={} job_error_rate_ok={} job_p95_ok={} worker_terminal_evaluated={} worker_terminal_ok={}",
        thresholds.read_evaluated,
        thresholds.read_error_rate_ok,
        thresholds.read_p95_ok,
        thresholds.job_evaluated,
        thresholds.job_error_rate_ok,
        thresholds.job_p95_ok,
        thresholds.worker_terminal_evaluated,
        thresholds.worker_terminal_ok
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scenario_mode_flags_are_stable() {
        assert!(ScenarioMode::Read.includes_read());
        assert!(!ScenarioMode::Read.includes_jobs());
        assert!(!ScenarioMode::Jobs.includes_read());
        assert!(ScenarioMode::Jobs.includes_jobs());
        assert!(ScenarioMode::Full.includes_read());
        assert!(ScenarioMode::Full.includes_jobs());
    }

    #[test]
    fn percentile_uses_merged_histogram() {
        let histogram = BTreeMap::from([(10, 1), (25, 3), (50, 1)]);
        assert_eq!(percentile(&histogram, 0.95), 50);
        assert_eq!(percentile(&histogram, 0.50), 25);
    }

    #[test]
    fn duplicates_path_uses_repeated_filter_params() {
        let manifest = profile_manifest(DatasetProfile::Pr);
        let path = build_duplicates_path(&manifest);
        assert!(path.starts_with("/duplicates?"));
        assert!(path.contains("deck_filter%5B%5D=Deck+00"));
        assert!(path.contains("tag_filter%5B%5D=dup-cluster-000"));
    }
}
