use std::sync::Arc;

use crate::error::ProviderError;
use crate::middleware::{Middleware, Next};
use crate::provider::Provider;
use crate::types::{Request, Response, StreamResponse};

pub struct AlpineClient {
    provider: Arc<dyn Provider>,
    middleware: Vec<Arc<dyn Middleware>>,
}

impl AlpineClient {
    pub fn new(provider: impl Provider + 'static) -> Self {
        Self {
            provider: Arc::new(provider),
            middleware: Vec::new(),
        }
    }

    pub fn with_middleware(mut self, m: impl Middleware + 'static) -> Self {
        self.middleware.push(Arc::new(m));
        self
    }

    /// Run the request through the middleware chain, then the provider.
    pub async fn complete(&self, req: Request) -> Result<Response, ProviderError> {
        let provider = Arc::clone(&self.provider);

        let core: Next = Box::new(move |r| Box::pin(async move { provider.complete(&r).await }));

        let chain = self.middleware.iter().rev().fold(core, |next, mw| {
            let mw = Arc::clone(mw);
            Box::new(move |r| mw.handle(r, next))
        });

        chain(req).await
    }

    /// Stream bypasses middleware for now — middleware is request/response
    /// oriented. Streaming middleware is a separate concern.
    pub async fn stream(&self, req: &Request) -> Result<StreamResponse<'_>, ProviderError> {
        self.provider.stream(req).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::ProviderError;
    use crate::middleware::Middleware;
    use crate::provider::Provider;
    use crate::types::*;
    use async_trait::async_trait;
    use futures::StreamExt;
    use std::future::Future;
    use std::pin::Pin;
    use std::time::Duration;

    // -- Stub provider ---------------------------------------------------------

    struct StubProvider {
        model: ModelId,
        content: String,
    }

    impl StubProvider {
        fn new(content: &str) -> Self {
            Self {
                model: ModelId::new("stub"),
                content: content.into(),
            }
        }
    }

    #[async_trait]
    impl Provider for StubProvider {
        async fn complete(&self, _req: &Request) -> Result<Response, ProviderError> {
            Ok(Response {
                content: self.content.clone(),
                tool_calls: vec![],
                usage: Usage {
                    input_tokens: 1,
                    output_tokens: 2,
                },
                model: self.model.clone(),
                finish_reason: FinishReason::Stop,
                latency: Duration::ZERO,
                raw: serde_json::Value::Null,
            })
        }

        async fn stream(&self, _req: &Request) -> Result<StreamResponse<'_>, ProviderError> {
            let chunks = vec![
                StreamChunk::Delta("hello".into()),
                StreamChunk::Done { usage: None },
            ];
            Ok(Box::pin(futures::stream::iter(chunks)))
        }

        fn model_id(&self) -> &ModelId {
            &self.model
        }
    }

    // -- Stub middleware --------------------------------------------------------

    struct AppendMiddleware {
        suffix: String,
    }

    impl AppendMiddleware {
        fn new(suffix: &str) -> Self {
            Self {
                suffix: suffix.into(),
            }
        }
    }

    impl Middleware for AppendMiddleware {
        fn handle(
            self: Arc<Self>,
            req: Request,
            next: crate::middleware::Next,
        ) -> Pin<Box<dyn Future<Output = Result<Response, ProviderError>> + Send>> {
            Box::pin(async move {
                let mut resp = next(req).await?;
                resp.content.push_str(&self.suffix);
                Ok(resp)
            })
        }
    }

    struct ErrorMiddleware;

    impl Middleware for ErrorMiddleware {
        fn handle(
            self: Arc<Self>,
            _req: Request,
            _next: crate::middleware::Next,
        ) -> Pin<Box<dyn Future<Output = Result<Response, ProviderError>> + Send>> {
            Box::pin(async { Err(ProviderError::Other("middleware error".into())) })
        }
    }

    // -- Tests -----------------------------------------------------------------

    #[tokio::test]
    async fn client_new() {
        let _client = AlpineClient::new(StubProvider::new("x"));
    }

    #[tokio::test]
    async fn complete_no_middleware() {
        let client = AlpineClient::new(StubProvider::new("hello"));
        let resp = client.complete(Request::default()).await.unwrap();
        assert_eq!(resp.content, "hello");
        assert_eq!(resp.usage.input_tokens, 1);
        assert_eq!(resp.usage.output_tokens, 2);
    }

    #[tokio::test]
    async fn complete_with_one_middleware() {
        let client = AlpineClient::new(StubProvider::new("base"))
            .with_middleware(AppendMiddleware::new(" [m1]"));
        let resp = client.complete(Request::default()).await.unwrap();
        assert_eq!(resp.content, "base [m1]");
    }

    #[tokio::test]
    async fn complete_with_two_middleware() {
        let client = AlpineClient::new(StubProvider::new("base"))
            .with_middleware(AppendMiddleware::new(" [first]"))
            .with_middleware(AppendMiddleware::new(" [second]"));
        let resp = client.complete(Request::default()).await.unwrap();
        // Middleware wraps onion-style: first added is outermost.
        // Inner (second) runs first on response, then outer (first).
        assert_eq!(resp.content, "base [second] [first]");
    }

    #[tokio::test]
    async fn complete_middleware_error() {
        let client = AlpineClient::new(StubProvider::new("x")).with_middleware(ErrorMiddleware);
        let err = client.complete(Request::default()).await.unwrap_err();
        assert!(err.to_string().contains("middleware error"));
    }

    #[tokio::test]
    async fn stream_bypasses_middleware() {
        let client = AlpineClient::new(StubProvider::new("x"))
            .with_middleware(AppendMiddleware::new(" [mod]"));
        let mut stream = client.stream(&Request::default()).await.unwrap();

        let first = stream.next().await.unwrap();
        match first {
            StreamChunk::Delta(text) => assert_eq!(text, "hello"),
            other => panic!("expected Delta, got: {other:?}"),
        }

        let second = stream.next().await.unwrap();
        match second {
            StreamChunk::Done { .. } => {}
            other => panic!("expected Done, got: {other:?}"),
        }

        assert!(stream.next().await.is_none());
    }
}
