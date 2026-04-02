use std::collections::BTreeMap;

use anyhow::Result;
use goose::metrics::{GooseMetrics, GooseRequestMetricAggregate};
use serde::Serialize;

use crate::context;

#[derive(Debug, Serialize)]
pub(crate) struct PerfReport {
    pub(crate) profile: String,
    pub(crate) scenario: String,
    pub(crate) base_url: String,
    pub(crate) report_only: bool,
    pub(crate) duration_secs: usize,
    pub(crate) total_users: usize,
    pub(crate) requests_seen: usize,
    pub(crate) read: RequestGroupSummary,
    pub(crate) jobs: RequestGroupSummary,
    pub(crate) worker_terminal_ratio: f64,
    pub(crate) worker_terminal_attempts: usize,
    pub(crate) thresholds: ThresholdReport,
}

#[derive(Debug, Serialize, Default)]
pub(crate) struct RequestGroupSummary {
    pub(crate) requests: usize,
    pub(crate) successes: usize,
    pub(crate) failures: usize,
    pub(crate) error_rate: f64,
    pub(crate) p95_ms: usize,
}

#[derive(Debug, Serialize)]
pub(crate) struct ThresholdReport {
    pub(crate) read_evaluated: bool,
    pub(crate) read_error_rate_ok: bool,
    pub(crate) read_p95_ok: bool,
    pub(crate) job_evaluated: bool,
    pub(crate) job_error_rate_ok: bool,
    pub(crate) job_p95_ok: bool,
    pub(crate) worker_terminal_evaluated: bool,
    pub(crate) worker_terminal_ok: bool,
}

pub(crate) fn build_report(metrics: &GooseMetrics) -> PerfReport {
    use std::sync::atomic::Ordering;

    let ctx = context();
    let read = summarize_requests(
        metrics
            .requests
            .values()
            .filter(|request| request.path.starts_with("read_")),
    );
    let jobs = summarize_requests(
        metrics
            .requests
            .values()
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

pub(crate) fn summarize_requests<'a>(
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

pub(crate) fn percentile(histogram: &BTreeMap<usize, usize>, percentile: f64) -> usize {
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

pub(crate) fn print_report(report: &PerfReport) {
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

pub(crate) fn enforce_thresholds(report: &PerfReport) -> Result<()> {
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
