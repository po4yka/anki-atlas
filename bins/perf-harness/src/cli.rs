use std::env;
use std::path::PathBuf;

use anyhow::{Context, Result};
use perf_support::DatasetProfile;

#[derive(Debug, Clone)]
pub(crate) struct Cli {
    pub(crate) profile: DatasetProfile,
    pub(crate) scenario: ScenarioMode,
    pub(crate) base_url: String,
    pub(crate) report_json: PathBuf,
    pub(crate) prepare_only: bool,
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum ScenarioMode {
    Read,
    Jobs,
    Full,
}

impl ScenarioMode {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Read => "read",
            Self::Jobs => "jobs",
            Self::Full => "full",
        }
    }

    pub(crate) fn includes_read(self) -> bool {
        matches!(self, Self::Read | Self::Full)
    }

    pub(crate) fn includes_jobs(self) -> bool {
        matches!(self, Self::Jobs | Self::Full)
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct LoadProfile {
    pub(crate) total_users: usize,
    pub(crate) hatch_rate: &'static str,
    pub(crate) run_time_secs: usize,
    pub(crate) read_p95_ms: usize,
    pub(crate) job_p95_ms: usize,
    pub(crate) terminal_ratio_min: f64,
    pub(crate) terminal_sla_secs: usize,
    pub(crate) report_only: bool,
}

impl LoadProfile {
    pub(crate) fn for_profile(profile: DatasetProfile) -> Self {
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

pub(crate) fn parse_cli() -> Result<Cli> {
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

pub(crate) fn print_usage() {
    println!(
        "perf-harness --profile <pr|nightly> --scenario <read|jobs|full> [--base-url URL] [--report-json FILE] [--prepare-only]"
    );
}

pub(crate) fn manifest_path_for(profile: DatasetProfile) -> std::path::PathBuf {
    std::path::PathBuf::from(format!(
        "target/perf/seed-manifest-{}.json",
        profile.as_str()
    ))
}
