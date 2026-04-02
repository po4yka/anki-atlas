use std::fs;
use std::sync::OnceLock;
use std::sync::atomic::AtomicUsize;
use std::time::Duration;

use anyhow::{Context, Result};
use goose::config::{GooseConfiguration, GooseDefault};
use goose::prelude::*;
use gumdrop::Options;
use perf_support::{manifest_json, profile_manifest, reset_and_seed};

mod cli;
mod http;
mod jobs;
mod report;
mod transactions;

use cli::{LoadProfile, ScenarioMode, manifest_path_for, parse_cli};
use report::{build_report, enforce_thresholds, print_report};
use transactions::{
    LoadContext, duplicates_request, enqueue_index_job, enqueue_sync_job, job_cancel_request,
    job_status_request, prime_job, search_filtered_request, search_request, search_rerank_request,
    topic_coverage_request, topic_gaps_request, topic_weak_notes_request, topics_request,
};

static CONTEXT: OnceLock<LoadContext> = OnceLock::new();

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

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use perf_support::DatasetProfile;

    use super::*;
    use crate::cli::ScenarioMode;
    use crate::report::percentile;
    use crate::transactions::build_duplicates_path;

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
