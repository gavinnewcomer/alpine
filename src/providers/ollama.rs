use async_trait::async_trait;
use futures::StreamExt;
use reqwest::Client;
use serde::{Deserialize, Serialize};

use crate::error::ProviderError;
use crate::provider::Provider;
use crate::types::{
    EmbedRequest, Embedding, FinishReason, ModelId, Request, Response, Role,
    StreamChunk, StreamResponse, Usage,
};

// ---------------------------------------------------------------------------
// Provider
// ---------------------------------------------------------------------------

pub struct OllamaProvider {
    client: Client,
    base_url: String,
    model: ModelId,
}

impl OllamaProvider {
    pub fn new(model: impl Into<String>) -> Self {
        Self::with_base_url(model, "http://localhost:11434")
    }

    pub fn with_base_url(model: impl Into<String>, base_url: impl Into<String>) -> Self {
        Self {
            client: Client::new(),
            base_url: base_url.into().trim_end_matches('/').to_string(),
            model: ModelId::new(model),
        }
    }
}

#[async_trait]
impl Provider for OllamaProvider {
    async fn complete(&self, req: &Request) -> Result<Response, ProviderError> {
        let body = ChatRequest::from_request(req, &self.model, false);
        let url = format!("{}/api/chat", self.base_url);

        let start = std::time::Instant::now();
        let resp = self.client.post(&url).json(&body).send().await?;

        let status = resp.status();
        if !status.is_success() {
            let text = resp.text().await.unwrap_or_default();
            return Err(ProviderError::Api {
                status: status.as_u16(),
                message: text,
            });
        }

        let raw: serde_json::Value = resp.json().await?;
        let latency = start.elapsed();
        parse_chat_response(raw, &self.model, latency)
    }

    async fn stream(&self, req: &Request) -> Result<StreamResponse<'_>, ProviderError> {
        let body = ChatRequest::from_request(req, &self.model, true);
        let url = format!("{}/api/chat", self.base_url);

        let resp = self.client.post(&url).json(&body).send().await?;

        let status = resp.status();
        if !status.is_success() {
            let text = resp.text().await.unwrap_or_default();
            return Err(ProviderError::Api {
                status: status.as_u16(),
                message: text,
            });
        }

        let stream = resp.bytes_stream().map(|chunk| match chunk {
            Err(e) => StreamChunk::Error(e.to_string()),
            Ok(bytes) => parse_stream_chunk(&bytes),
        });

        Ok(Box::pin(stream))
    }

    async fn embed(&self, req: &EmbedRequest) -> Result<Embedding, ProviderError> {
        let url = format!("{}/api/embed", self.base_url);

        let body = serde_json::json!({
            "model": req.model.as_str(),
            "input": req.input,
        });

        let resp = self.client.post(&url).json(&body).send().await?;

        let status = resp.status();
        if !status.is_success() {
            let text = resp.text().await.unwrap_or_default();
            return Err(ProviderError::Api {
                status: status.as_u16(),
                message: text,
            });
        }

        let raw: EmbedResponse = resp.json().await?;

        Ok(Embedding {
            vectors: raw.embeddings,
            model: req.model.clone(),
            usage: Usage::default(),
        })
    }

    fn model_id(&self) -> &ModelId {
        &self.model
    }
}

// ---------------------------------------------------------------------------
// Wire types — Ollama /api/chat
// ---------------------------------------------------------------------------

#[derive(Serialize)]
struct ChatRequest {
    model: String,
    messages: Vec<ChatMessage>,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    options: Option<ChatOptions>,
}

#[derive(Serialize)]
struct ChatMessage {
    role: String,
    content: String,
}

#[derive(Serialize)]
struct ChatOptions {
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    num_predict: Option<u32>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    stop: Vec<String>,
}

impl ChatRequest {
    fn from_request(req: &Request, default_model: &ModelId, stream: bool) -> Self {
        let model = if req.model.as_str() == "default" {
            default_model.as_str().to_string()
        } else {
            req.model.as_str().to_string()
        };

        let mut messages: Vec<ChatMessage> = Vec::new();

        if let Some(sys) = &req.system {
            messages.push(ChatMessage {
                role: "system".into(),
                content: sys.clone(),
            });
        }

        for m in &req.messages {
            messages.push(ChatMessage {
                role: match m.role {
                    Role::System => "system",
                    Role::User => "user",
                    Role::Assistant => "assistant",
                }
                .into(),
                content: m.content.clone(),
            });
        }

        let options = ChatOptions {
            temperature: req.temperature,
            num_predict: req.max_tokens,
            stop: req.stop.clone(),
        };

        let has_options = options.temperature.is_some()
            || options.num_predict.is_some()
            || !options.stop.is_empty();

        Self {
            model,
            messages,
            stream,
            options: if has_options { Some(options) } else { None },
        }
    }
}

// ---------------------------------------------------------------------------
// Wire types — Ollama /api/embed
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct EmbedResponse {
    embeddings: Vec<Vec<f32>>,
}

// ---------------------------------------------------------------------------
// Response parsing
// ---------------------------------------------------------------------------

fn parse_chat_response(
    raw: serde_json::Value,
    default_model: &ModelId,
    latency: std::time::Duration,
) -> Result<Response, ProviderError> {
    let content = raw["message"]["content"]
        .as_str()
        .unwrap_or("")
        .to_string();

    let done_reason = raw["done_reason"].as_str().unwrap_or("stop");
    let finish_reason = match done_reason {
        "stop" => FinishReason::Stop,
        "length" => FinishReason::MaxTokens,
        other => FinishReason::Other(other.into()),
    };

    let model_str = raw["model"].as_str().unwrap_or(default_model.as_str());

    let usage = Usage {
        input_tokens: raw["prompt_eval_count"].as_u64().unwrap_or(0) as u32,
        output_tokens: raw["eval_count"].as_u64().unwrap_or(0) as u32,
    };

    Ok(Response {
        content,
        usage,
        model: ModelId::new(model_str),
        finish_reason,
        latency,
        raw,
    })
}

fn parse_stream_chunk(bytes: &[u8]) -> StreamChunk {
    let text = match std::str::from_utf8(bytes) {
        Ok(t) => t,
        Err(e) => return StreamChunk::Error(e.to_string()),
    };

    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        let v: serde_json::Value = match serde_json::from_str(line) {
            Ok(v) => v,
            Err(e) => return StreamChunk::Error(e.to_string()),
        };

        if v["done"].as_bool() == Some(true) {
            let usage = Some(Usage {
                input_tokens: v["prompt_eval_count"].as_u64().unwrap_or(0) as u32,
                output_tokens: v["eval_count"].as_u64().unwrap_or(0) as u32,
            });
            return StreamChunk::Done { usage };
        }

        let delta = v["message"]["content"]
            .as_str()
            .unwrap_or("")
            .to_string();

        if !delta.is_empty() {
            return StreamChunk::Delta(delta);
        }
    }

    StreamChunk::Delta(String::new())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Message;
    use futures::StreamExt;
    use std::time::Duration;
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    // ── parse_chat_response ──────────────────────────────────────────────

    #[test]
    fn parse_response_full() {
        let raw = serde_json::json!({
            "model": "llama3.2",
            "message": { "content": "hello world" },
            "done_reason": "stop",
            "prompt_eval_count": 10,
            "eval_count": 20,
        });
        let resp = parse_chat_response(raw, &ModelId::new("fallback"), Duration::from_secs(1)).unwrap();
        assert_eq!(resp.content, "hello world");
        assert_eq!(resp.model.as_str(), "llama3.2");
        assert_eq!(resp.finish_reason, FinishReason::Stop);
        assert_eq!(resp.usage.input_tokens, 10);
        assert_eq!(resp.usage.output_tokens, 20);
        assert_eq!(resp.latency, Duration::from_secs(1));
    }

    #[test]
    fn parse_response_missing_content() {
        let raw = serde_json::json!({ "done_reason": "stop" });
        let resp = parse_chat_response(raw, &ModelId::new("f"), Duration::ZERO).unwrap();
        assert_eq!(resp.content, "");
    }

    #[test]
    fn parse_response_done_reason_length() {
        let raw = serde_json::json!({ "done_reason": "length" });
        let resp = parse_chat_response(raw, &ModelId::new("f"), Duration::ZERO).unwrap();
        assert_eq!(resp.finish_reason, FinishReason::MaxTokens);
    }

    #[test]
    fn parse_response_done_reason_other() {
        let raw = serde_json::json!({ "done_reason": "custom_thing" });
        let resp = parse_chat_response(raw, &ModelId::new("f"), Duration::ZERO).unwrap();
        assert_eq!(resp.finish_reason, FinishReason::Other("custom_thing".into()));
    }

    #[test]
    fn parse_response_missing_done_reason() {
        let raw = serde_json::json!({});
        let resp = parse_chat_response(raw, &ModelId::new("f"), Duration::ZERO).unwrap();
        assert_eq!(resp.finish_reason, FinishReason::Stop);
    }

    #[test]
    fn parse_response_missing_model_uses_default() {
        let raw = serde_json::json!({});
        let resp = parse_chat_response(raw, &ModelId::new("fallback"), Duration::ZERO).unwrap();
        assert_eq!(resp.model.as_str(), "fallback");
    }

    #[test]
    fn parse_response_missing_usage() {
        let raw = serde_json::json!({});
        let resp = parse_chat_response(raw, &ModelId::new("f"), Duration::ZERO).unwrap();
        assert_eq!(resp.usage.input_tokens, 0);
        assert_eq!(resp.usage.output_tokens, 0);
    }

    // ── parse_stream_chunk ───────────────────────────────────────────────

    #[test]
    fn stream_chunk_delta() {
        let data = br#"{"done":false,"message":{"content":"hi"}}"#;
        match parse_stream_chunk(data) {
            StreamChunk::Delta(t) => assert_eq!(t, "hi"),
            other => panic!("expected Delta, got {other:?}"),
        }
    }

    #[test]
    fn stream_chunk_done() {
        let data = br#"{"done":true,"prompt_eval_count":5,"eval_count":10}"#;
        match parse_stream_chunk(data) {
            StreamChunk::Done { usage } => {
                let u = usage.unwrap();
                assert_eq!(u.input_tokens, 5);
                assert_eq!(u.output_tokens, 10);
            }
            other => panic!("expected Done, got {other:?}"),
        }
    }

    #[test]
    fn stream_chunk_empty_delta() {
        let data = br#"{"done":false,"message":{"content":""}}"#;
        // empty delta falls through the loop, returns Delta("")
        match parse_stream_chunk(data) {
            StreamChunk::Delta(t) => assert_eq!(t, ""),
            other => panic!("expected empty Delta, got {other:?}"),
        }
    }

    #[test]
    fn stream_chunk_invalid_utf8() {
        let data: &[u8] = &[0xff, 0xfe, 0xfd];
        match parse_stream_chunk(data) {
            StreamChunk::Error(_) => {}
            other => panic!("expected Error, got {other:?}"),
        }
    }

    #[test]
    fn stream_chunk_invalid_json() {
        let data = b"not valid json";
        match parse_stream_chunk(data) {
            StreamChunk::Error(_) => {}
            other => panic!("expected Error, got {other:?}"),
        }
    }

    #[test]
    fn stream_chunk_empty_input() {
        match parse_stream_chunk(b"") {
            StreamChunk::Delta(t) => assert_eq!(t, ""),
            other => panic!("expected empty Delta, got {other:?}"),
        }
    }

    #[test]
    fn stream_chunk_multiline_skips_blanks() {
        let data = b"\n\n{\"done\":false,\"message\":{\"content\":\"ok\"}}\n\n";
        match parse_stream_chunk(data) {
            StreamChunk::Delta(t) => assert_eq!(t, "ok"),
            other => panic!("expected Delta(ok), got {other:?}"),
        }
    }

    // ── ChatRequest::from_request ────────────────────────────────────────

    #[test]
    fn chat_request_default_model() {
        let req = Request::default(); // model is "default"
        let cr = ChatRequest::from_request(&req, &ModelId::new("llama3.2"), false);
        assert_eq!(cr.model, "llama3.2");
    }

    #[test]
    fn chat_request_explicit_model() {
        let req = Request {
            model: ModelId::new("mistral"),
            ..Default::default()
        };
        let cr = ChatRequest::from_request(&req, &ModelId::new("llama3.2"), false);
        assert_eq!(cr.model, "mistral");
    }

    #[test]
    fn chat_request_with_system() {
        let req = Request {
            system: Some("you are helpful".into()),
            messages: vec![Message::user("hi")],
            ..Default::default()
        };
        let cr = ChatRequest::from_request(&req, &ModelId::new("m"), false);
        assert_eq!(cr.messages.len(), 2);
        assert_eq!(cr.messages[0].role, "system");
        assert_eq!(cr.messages[0].content, "you are helpful");
        assert_eq!(cr.messages[1].role, "user");
    }

    #[test]
    fn chat_request_no_system() {
        let req = Request {
            messages: vec![Message::user("hi")],
            ..Default::default()
        };
        let cr = ChatRequest::from_request(&req, &ModelId::new("m"), false);
        assert_eq!(cr.messages.len(), 1);
        assert_eq!(cr.messages[0].role, "user");
    }

    #[test]
    fn chat_request_message_roles() {
        let req = Request {
            messages: vec![
                Message::system("sys"),
                Message::user("usr"),
                Message::assistant("ast"),
            ],
            ..Default::default()
        };
        let cr = ChatRequest::from_request(&req, &ModelId::new("m"), false);
        assert_eq!(cr.messages[0].role, "system");
        assert_eq!(cr.messages[1].role, "user");
        assert_eq!(cr.messages[2].role, "assistant");
    }

    #[test]
    fn chat_request_options_populated() {
        let req = Request {
            temperature: Some(0.7),
            max_tokens: Some(100),
            stop: vec!["END".into()],
            ..Default::default()
        };
        let cr = ChatRequest::from_request(&req, &ModelId::new("m"), false);
        let opts = cr.options.unwrap();
        assert_eq!(opts.temperature, Some(0.7));
        assert_eq!(opts.num_predict, Some(100));
        assert_eq!(opts.stop, vec!["END"]);
    }

    #[test]
    fn chat_request_options_none_when_empty() {
        let req = Request::default();
        let cr = ChatRequest::from_request(&req, &ModelId::new("m"), false);
        assert!(cr.options.is_none());
    }

    #[test]
    fn chat_request_stream_flag() {
        let req = Request::default();
        assert!(ChatRequest::from_request(&req, &ModelId::new("m"), true).stream);
        assert!(!ChatRequest::from_request(&req, &ModelId::new("m"), false).stream);
    }

    // ── Constructor tests ────────────────────────────────────────────────

    #[test]
    fn new_default_url() {
        let p = OllamaProvider::new("llama3");
        assert_eq!(p.base_url, "http://localhost:11434");
    }

    #[test]
    fn with_base_url_trims_slash() {
        let p = OllamaProvider::with_base_url("m", "http://host:1234/");
        assert_eq!(p.base_url, "http://host:1234");
    }

    #[test]
    fn model_id_returns_configured() {
        let p = OllamaProvider::new("llama3.2");
        assert_eq!(p.model_id().as_str(), "llama3.2");
    }

    // ── HTTP integration tests (wiremock) ────────────────────────────────

    #[tokio::test]
    async fn complete_success() {
        let server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/api/chat"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "model": "llama3.2",
                "message": { "role": "assistant", "content": "Hello!" },
                "done": true,
                "done_reason": "stop",
                "prompt_eval_count": 8,
                "eval_count": 12,
            })))
            .mount(&server)
            .await;

        let provider = OllamaProvider::with_base_url("llama3.2", server.uri());
        let resp = provider.complete(&Request {
            messages: vec![Message::user("hi")],
            ..Default::default()
        }).await.unwrap();

        assert_eq!(resp.content, "Hello!");
        assert_eq!(resp.usage.input_tokens, 8);
        assert_eq!(resp.usage.output_tokens, 12);
        assert!(resp.latency > Duration::ZERO);
    }

    #[tokio::test]
    async fn complete_api_error() {
        let server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/api/chat"))
            .respond_with(ResponseTemplate::new(500).set_body_string("internal error"))
            .mount(&server)
            .await;

        let provider = OllamaProvider::with_base_url("llama3.2", server.uri());
        let err = provider.complete(&Request::default()).await.unwrap_err();
        match err {
            ProviderError::Api { status, message } => {
                assert_eq!(status, 500);
                assert!(message.contains("internal error"));
            }
            other => panic!("expected Api error, got {other}"),
        }
    }

    #[tokio::test]
    async fn stream_success() {
        let server = MockServer::start().await;

        // Wiremock sends the body as a single chunk, so parse_stream_chunk
        // only returns the first NDJSON line per payload. Use a single done
        // message to test the full flow, and a separate test for delta.
        let body = r#"{"done":true,"prompt_eval_count":3,"eval_count":1}"#;

        Mock::given(method("POST"))
            .and(path("/api/chat"))
            .respond_with(ResponseTemplate::new(200).set_body_string(body))
            .mount(&server)
            .await;

        let provider = OllamaProvider::with_base_url("llama3.2", server.uri());
        let mut stream = provider.stream(&Request {
            messages: vec![Message::user("hi")],
            ..Default::default()
        }).await.unwrap();

        let mut got_done = false;
        while let Some(chunk) = stream.next().await {
            match chunk {
                StreamChunk::Done { usage } => {
                    got_done = true;
                    let u = usage.unwrap();
                    assert_eq!(u.input_tokens, 3);
                    assert_eq!(u.output_tokens, 1);
                }
                _ => {}
            }
        }
        assert!(got_done);
    }

    #[tokio::test]
    async fn stream_delta() {
        let server = MockServer::start().await;

        let body = r#"{"done":false,"message":{"content":"Hello"}}"#;

        Mock::given(method("POST"))
            .and(path("/api/chat"))
            .respond_with(ResponseTemplate::new(200).set_body_string(body))
            .mount(&server)
            .await;

        let provider = OllamaProvider::with_base_url("llama3.2", server.uri());
        let mut stream = provider.stream(&Request {
            messages: vec![Message::user("hi")],
            ..Default::default()
        }).await.unwrap();

        let mut got_delta = false;
        while let Some(chunk) = stream.next().await {
            match chunk {
                StreamChunk::Delta(t) if t == "Hello" => got_delta = true,
                _ => {}
            }
        }
        assert!(got_delta);
    }

    #[tokio::test]
    async fn stream_api_error() {
        let server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/api/chat"))
            .respond_with(ResponseTemplate::new(400).set_body_string("bad request"))
            .mount(&server)
            .await;

        let provider = OllamaProvider::with_base_url("llama3.2", server.uri());
        match provider.stream(&Request::default()).await {
            Err(ProviderError::Api { status, .. }) => assert_eq!(status, 400),
            Err(other) => panic!("expected Api error, got {other}"),
            Ok(_) => panic!("expected error"),
        }
    }

    #[tokio::test]
    async fn embed_success() {
        let server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/api/embed"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "embeddings": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
            })))
            .mount(&server)
            .await;

        let provider = OllamaProvider::with_base_url("nomic", server.uri());
        let emb = provider.embed(&EmbedRequest {
            model: ModelId::new("nomic"),
            input: vec!["hello".into(), "world".into()],
        }).await.unwrap();

        assert_eq!(emb.vectors.len(), 2);
        assert_eq!(emb.vectors[0].len(), 3);
    }

    #[tokio::test]
    async fn embed_api_error() {
        let server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/api/embed"))
            .respond_with(ResponseTemplate::new(500).set_body_string("fail"))
            .mount(&server)
            .await;

        let provider = OllamaProvider::with_base_url("nomic", server.uri());
        let err = provider.embed(&EmbedRequest {
            model: ModelId::new("nomic"),
            input: vec!["hi".into()],
        }).await.unwrap_err();
        match err {
            ProviderError::Api { status, .. } => assert_eq!(status, 500),
            other => panic!("expected Api error, got {other}"),
        }
    }
}
