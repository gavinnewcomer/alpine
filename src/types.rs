use std::time::Duration;

use futures::stream::BoxStream;
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// ModelId
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ModelId(pub String);

impl ModelId {
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl Default for ModelId {
    fn default() -> Self {
        Self("default".into())
    }
}

impl std::fmt::Display for ModelId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

// ---------------------------------------------------------------------------
// Message
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: Role,
    pub content: String,
}

impl Message {
    pub fn user(content: impl Into<String>) -> Self {
        Self { role: Role::User, content: content.into() }
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self { role: Role::Assistant, content: content.into() }
    }

    pub fn system(content: impl Into<String>) -> Self {
        Self { role: Role::System, content: content.into() }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
}

// ---------------------------------------------------------------------------
// Request / Response
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Default)]
pub struct Request {
    pub messages: Vec<Message>,
    pub model: ModelId,
    pub max_tokens: Option<u32>,
    pub temperature: Option<f32>,
    pub system: Option<String>,
    pub stop: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct Response {
    pub content: String,
    pub usage: Usage,
    pub model: ModelId,
    pub finish_reason: FinishReason,
    pub latency: Duration,
    pub raw: serde_json::Value,
}

#[derive(Debug, Clone, Default)]
pub struct Usage {
    pub input_tokens: u32,
    pub output_tokens: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FinishReason {
    Stop,
    MaxTokens,
    ContentFilter,
    Other(String),
}

impl Default for FinishReason {
    fn default() -> Self {
        Self::Stop
    }
}

impl std::fmt::Display for Response {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "[{}] ({:.0?})", self.model, self.latency)?;
        writeln!(f, "{}", self.content)?;
        write!(
            f,
            "tokens: {} in / {} out | finish: {:?}",
            self.usage.input_tokens, self.usage.output_tokens, self.finish_reason
        )
    }
}

// ---------------------------------------------------------------------------
// Streaming
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub enum StreamChunk {
    Delta(String),
    Done { usage: Option<Usage> },
    Error(String),
}

/// Convenience alias used throughout the crate.
pub type StreamResponse<'a> = BoxStream<'a, StreamChunk>;

// ---------------------------------------------------------------------------
// Embeddings
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct EmbedRequest {
    pub model: ModelId,
    pub input: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct Embedding {
    pub vectors: Vec<Vec<f32>>,
    pub model: ModelId,
    pub usage: Usage,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;
    use std::time::Duration;

    // -- ModelId ---------------------------------------------------------------

    #[test]
    fn model_id_new_and_as_str() {
        let m = ModelId::new("gpt-4");
        assert_eq!(m.as_str(), "gpt-4");
    }

    #[test]
    fn model_id_default() {
        assert_eq!(ModelId::default().as_str(), "default");
    }

    #[test]
    fn model_id_display() {
        let m = ModelId::new("claude-3");
        assert_eq!(format!("{m}"), "claude-3");
    }

    #[test]
    fn model_id_eq_and_hash() {
        let a = ModelId::new("x");
        let b = ModelId::new("x");
        let c = ModelId::new("y");
        assert_eq!(a, b);
        assert_ne!(a, c);

        let mut set = HashSet::new();
        set.insert(a);
        set.insert(b);
        assert_eq!(set.len(), 1);
    }

    #[test]
    fn model_id_serde_roundtrip() {
        let m = ModelId::new("llama3.2");
        let json = serde_json::to_string(&m).unwrap();
        let back: ModelId = serde_json::from_str(&json).unwrap();
        assert_eq!(m, back);
    }

    // -- Message ---------------------------------------------------------------

    #[test]
    fn message_user() {
        let m = Message::user("hi");
        assert_eq!(m.role, Role::User);
        assert_eq!(m.content, "hi");
    }

    #[test]
    fn message_assistant() {
        let m = Message::assistant("ok");
        assert_eq!(m.role, Role::Assistant);
        assert_eq!(m.content, "ok");
    }

    #[test]
    fn message_system() {
        let m = Message::system("you are helpful");
        assert_eq!(m.role, Role::System);
        assert_eq!(m.content, "you are helpful");
    }

    // -- Role ------------------------------------------------------------------

    #[test]
    fn role_serde_roundtrip() {
        for (role, expected) in [
            (Role::User, "\"user\""),
            (Role::Assistant, "\"assistant\""),
            (Role::System, "\"system\""),
        ] {
            let json = serde_json::to_string(&role).unwrap();
            assert_eq!(json, expected);
            let back: Role = serde_json::from_str(&json).unwrap();
            assert_eq!(back, role);
        }
    }

    // -- Request ---------------------------------------------------------------

    #[test]
    fn request_default() {
        let r = Request::default();
        assert!(r.messages.is_empty());
        assert_eq!(r.model, ModelId::default());
        assert!(r.max_tokens.is_none());
        assert!(r.temperature.is_none());
        assert!(r.system.is_none());
        assert!(r.stop.is_empty());
    }

    // -- Response Display ------------------------------------------------------

    #[test]
    fn response_display() {
        let resp = Response {
            content: "Hello!".into(),
            usage: Usage { input_tokens: 10, output_tokens: 5 },
            model: ModelId::new("test-model"),
            finish_reason: FinishReason::Stop,
            latency: Duration::from_millis(1234),
            raw: serde_json::Value::Null,
        };
        let s = format!("{resp}");
        assert!(s.contains("test-model"));
        assert!(s.contains("Hello!"));
        assert!(s.contains("10 in"));
        assert!(s.contains("5 out"));
        assert!(s.contains("Stop"));
        // latency formatted with {:.0?} — should contain "1.234s" or "1234ms"
        assert!(s.contains("1"));
    }

    // -- FinishReason ----------------------------------------------------------

    #[test]
    fn finish_reason_default_is_stop() {
        assert_eq!(FinishReason::default(), FinishReason::Stop);
    }

    #[test]
    fn finish_reason_variants() {
        assert_eq!(FinishReason::Stop, FinishReason::Stop);
        assert_ne!(FinishReason::Stop, FinishReason::MaxTokens);
        assert_ne!(FinishReason::MaxTokens, FinishReason::ContentFilter);
        let other = FinishReason::Other("custom".into());
        assert_eq!(other, FinishReason::Other("custom".into()));
        assert_ne!(other, FinishReason::Other("different".into()));
    }

    // -- Usage -----------------------------------------------------------------

    #[test]
    fn usage_default() {
        let u = Usage::default();
        assert_eq!(u.input_tokens, 0);
        assert_eq!(u.output_tokens, 0);
    }

    // -- StreamChunk -----------------------------------------------------------

    #[test]
    fn stream_chunk_debug() {
        let _ = format!("{:?}", StreamChunk::Delta("hi".into()));
        let _ = format!("{:?}", StreamChunk::Done { usage: None });
        let _ = format!("{:?}", StreamChunk::Done { usage: Some(Usage::default()) });
        let _ = format!("{:?}", StreamChunk::Error("err".into()));
    }

    // -- EmbedRequest / Embedding ----------------------------------------------

    #[test]
    fn embed_request_construction() {
        let r = EmbedRequest {
            model: ModelId::new("nomic"),
            input: vec!["hello".into(), "world".into()],
        };
        assert_eq!(r.model.as_str(), "nomic");
        assert_eq!(r.input.len(), 2);
    }

    #[test]
    fn embedding_construction() {
        let e = Embedding {
            vectors: vec![vec![0.1, 0.2], vec![0.3, 0.4]],
            model: ModelId::new("nomic"),
            usage: Usage { input_tokens: 4, output_tokens: 0 },
        };
        assert_eq!(e.vectors.len(), 2);
        assert_eq!(e.vectors[0].len(), 2);
        assert_eq!(e.model.as_str(), "nomic");
    }
}
