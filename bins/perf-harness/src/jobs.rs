use std::time::Duration;

use goose::prelude::*;
use serde::Deserialize;

use crate::context;
use crate::http::{ensure_success_response, post_json_named};

#[derive(Clone)]
pub(crate) struct JobSession {
    pub(crate) last_job_id: Option<String>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct JobAcceptedResponse {
    pub(crate) job_id: String,
}

#[derive(Debug, Deserialize)]
pub(crate) struct JobStatusResponse {
    pub(crate) status: String,
}

pub(crate) async fn enqueue_job(
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

pub(crate) async fn poll_until_terminal(
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
