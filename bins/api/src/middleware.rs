use axum::body::Body;
use axum::http::{HeaderValue, Request, StatusCode};
use axum::response::Response;
use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll};
use tower::{Layer, Service};
use uuid::Uuid;

// --- Correlation ID ---

/// Layer that sets X-Request-ID on every response.
#[derive(Clone)]
pub struct CorrelationIdLayer;

impl<S> Layer<S> for CorrelationIdLayer {
    type Service = CorrelationIdService<S>;

    fn layer(&self, inner: S) -> Self::Service {
        CorrelationIdService { inner }
    }
}

#[derive(Clone)]
pub struct CorrelationIdService<S> {
    inner: S,
}

impl<S> Service<Request<Body>> for CorrelationIdService<S>
where
    S: Service<Request<Body>, Response = Response> + Clone + Send + 'static,
    S::Future: Send + 'static,
{
    type Response = S::Response;
    type Error = S::Error;
    type Future = Pin<Box<dyn Future<Output = Result<Self::Response, Self::Error>> + Send>>;

    fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        self.inner.poll_ready(cx)
    }

    fn call(&mut self, req: Request<Body>) -> Self::Future {
        let request_id = req
            .headers()
            .get("x-request-id")
            .and_then(|v| v.to_str().ok())
            .map(|s| s.to_string())
            .unwrap_or_else(|| Uuid::new_v4().to_string());

        let future = self.inner.call(req);
        Box::pin(async move {
            let mut resp = future.await?;
            resp.headers_mut().insert(
                "x-request-id",
                HeaderValue::from_str(&request_id).unwrap_or_else(|_| {
                    HeaderValue::from_str(&Uuid::new_v4().to_string()).unwrap()
                }),
            );
            Ok(resp)
        })
    }
}

// --- API Key ---

/// Layer that validates X-API-Key header if a key is configured.
#[derive(Clone)]
pub struct ApiKeyLayer {
    api_key: Option<String>,
}

impl ApiKeyLayer {
    pub fn new(api_key: Option<String>) -> Self {
        Self { api_key }
    }
}

impl<S> Layer<S> for ApiKeyLayer {
    type Service = ApiKeyService<S>;

    fn layer(&self, inner: S) -> Self::Service {
        ApiKeyService {
            inner,
            api_key: self.api_key.clone(),
        }
    }
}

#[derive(Clone)]
pub struct ApiKeyService<S> {
    inner: S,
    api_key: Option<String>,
}

impl<S> Service<Request<Body>> for ApiKeyService<S>
where
    S: Service<Request<Body>, Response = Response> + Clone + Send + 'static,
    S::Future: Send + 'static,
{
    type Response = S::Response;
    type Error = S::Error;
    type Future = Pin<Box<dyn Future<Output = Result<Self::Response, Self::Error>> + Send>>;

    fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        self.inner.poll_ready(cx)
    }

    fn call(&mut self, req: Request<Body>) -> Self::Future {
        if let Some(expected) = &self.api_key {
            let provided = req
                .headers()
                .get("x-api-key")
                .and_then(|v| v.to_str().ok());

            if provided != Some(expected.as_str()) {
                return Box::pin(async {
                    Ok(Response::builder()
                        .status(StatusCode::UNAUTHORIZED)
                        .body(Body::from("unauthorized"))
                        .unwrap())
                });
            }
        }

        let future = self.inner.call(req);
        Box::pin(async move { future.await })
    }
}
