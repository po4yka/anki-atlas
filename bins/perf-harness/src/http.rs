use goose::goose::GooseResponse;
use goose::prelude::*;

pub(crate) async fn post_json_named<T: serde::Serialize + ?Sized>(
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

pub(crate) async fn post_empty_named(
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

pub(crate) fn ensure_success_response(
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
