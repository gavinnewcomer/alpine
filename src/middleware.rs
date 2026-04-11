use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use crate::error::ProviderError;
use crate::types::{Request, Response};

/// The rest of the middleware chain + provider call.
pub type Next =
    Box<dyn FnOnce(Request) -> Pin<Box<dyn Future<Output = Result<Response, ProviderError>> + Send>> + Send>;

/// Middleware that can inspect / transform a request before it reaches the
/// provider and the response after it comes back.
pub trait Middleware: Send + Sync + 'static {
    fn handle(
        self: Arc<Self>,
        req: Request,
        next: Next,
    ) -> Pin<Box<dyn Future<Output = Result<Response, ProviderError>> + Send>>;
}
