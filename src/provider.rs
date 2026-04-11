use async_trait::async_trait;

use crate::error::ProviderError;
use crate::types::{EmbedRequest, Embedding, ModelId, Request, Response, StreamResponse};

#[async_trait]
pub trait Provider: Send + Sync {
    async fn complete(&self, req: &Request) -> Result<Response, ProviderError>;

    async fn stream(&self, req: &Request) -> Result<StreamResponse<'_>, ProviderError>;

    async fn embed(&self, req: &EmbedRequest) -> Result<Embedding, ProviderError> {
        let _ = req;
        Err(ProviderError::Unsupported(
            "embedding not supported by this provider".into(),
        ))
    }

    fn model_id(&self) -> &ModelId;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{EmbedRequest, FinishReason, ModelId, Usage};
    use std::time::Duration;

    struct StubProvider(ModelId);

    #[async_trait]
    impl Provider for StubProvider {
        async fn complete(&self, _req: &Request) -> Result<Response, ProviderError> {
            Ok(Response {
                content: "stub".into(),
                usage: Usage::default(),
                model: self.0.clone(),
                finish_reason: FinishReason::Stop,
                latency: Duration::ZERO,
                raw: serde_json::Value::Null,
            })
        }

        async fn stream(&self, _req: &Request) -> Result<StreamResponse<'_>, ProviderError> {
            Ok(Box::pin(futures::stream::empty()))
        }

        fn model_id(&self) -> &ModelId {
            &self.0
        }
    }

    #[tokio::test]
    async fn default_embed_returns_unsupported() {
        let p = StubProvider(ModelId::new("test"));
        let req = EmbedRequest {
            model: ModelId::new("test"),
            input: vec!["hi".into()],
        };
        let err = p.embed(&req).await.unwrap_err();
        match err {
            ProviderError::Unsupported(msg) => assert!(msg.contains("embedding")),
            other => panic!("expected Unsupported, got: {other}"),
        }
    }

    #[test]
    fn model_id_accessor() {
        let p = StubProvider(ModelId::new("my-model"));
        assert_eq!(p.model_id().as_str(), "my-model");
    }

    #[tokio::test]
    async fn stub_complete_and_stream() {
        let p = StubProvider(ModelId::new("test"));
        let req = Request::default();
        let resp = p.complete(&req).await.unwrap();
        assert_eq!(resp.content, "stub");

        let stream = p.stream(&req).await.unwrap();
        let chunks: Vec<_> = futures::StreamExt::collect(stream).await;
        assert!(chunks.is_empty());
    }
}
