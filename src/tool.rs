//! Tool execution: a `Tool` trait, a `Toolbox` registry, and the agentic loop
//! (`AlpineClient::run_tools`) that ties them to the model.
//!
//! The provider layer already knows how to advertise tools and parse the
//! model's `tool_use` requests. What was missing is the *execution* half: given
//! a set of callable tools, drive the call → execute → feed-result-back loop
//! until the model stops asking for tools. That loop lives here.
//!
//! ```no_run
//! # use alpine::{AlpineClient, Request, Message, ToolDefinition, Tool, ToolError, Toolbox};
//! # use async_trait::async_trait;
//! struct Echo;
//!
//! #[async_trait]
//! impl Tool for Echo {
//!     fn definition(&self) -> ToolDefinition {
//!         ToolDefinition::new(
//!             "echo",
//!             "Echo the given text back",
//!             serde_json::json!({
//!                 "type": "object",
//!                 "properties": { "text": { "type": "string" } },
//!                 "required": ["text"]
//!             }),
//!         )
//!     }
//!
//!     async fn call(&self, input: serde_json::Value) -> Result<String, ToolError> {
//!         Ok(input["text"].as_str().unwrap_or_default().to_string())
//!     }
//! }
//!
//! # async fn run(client: AlpineClient) -> Result<(), Box<dyn std::error::Error>> {
//! let tools = Toolbox::new().with(Echo);
//! let req = Request {
//!     messages: vec![Message::user("Echo the word 'hello'.")],
//!     ..Default::default()
//! };
//! let run = client.run_tools(req, &tools, 8).await?;
//! println!("{}", run.response.content);
//! # Ok(())
//! # }
//! ```

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use thiserror::Error;

use crate::client::AlpineClient;
use crate::error::ProviderError;
use crate::types::{Message, Request, Response, ToolDefinition};

/// An error raised while executing a tool. Returning this from [`Tool::call`]
/// does **not** abort the run — the error text is handed back to the model as an
/// error `tool_result` so it can recover or try a different approach.
#[derive(Debug, Error)]
#[error("{0}")]
pub struct ToolError(pub String);

impl From<String> for ToolError {
    fn from(s: String) -> Self {
        ToolError(s)
    }
}

impl From<&str> for ToolError {
    fn from(s: &str) -> Self {
        ToolError(s.to_string())
    }
}

/// A tool the model can call. Implementors advertise a [`ToolDefinition`] and
/// execute against the model-supplied JSON input.
#[async_trait]
pub trait Tool: Send + Sync {
    /// The provider-agnostic definition advertised to the model. The `name`
    /// here is what the model uses to call the tool and how the [`Toolbox`]
    /// keys it.
    fn definition(&self) -> ToolDefinition;

    /// Execute the tool with the model-supplied input. `Ok` content is returned
    /// to the model as the tool result; `Err` is returned as an *error* tool
    /// result (the loop keeps going so the model can react).
    async fn call(&self, input: serde_json::Value) -> Result<String, ToolError>;
}

/// A registry of tools, keyed by their advertised name.
#[derive(Default, Clone)]
pub struct Toolbox {
    tools: HashMap<String, Arc<dyn Tool>>,
}

impl Toolbox {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a tool, consuming and returning `self` for chaining. A tool
    /// whose name collides with an existing one replaces it.
    pub fn with(mut self, tool: impl Tool + 'static) -> Self {
        self.add(tool);
        self
    }

    /// Register a tool in place.
    pub fn add(&mut self, tool: impl Tool + 'static) {
        let name = tool.definition().name;
        self.tools.insert(name, Arc::new(tool));
    }

    /// The definitions advertised to the model for this toolbox.
    pub fn definitions(&self) -> Vec<ToolDefinition> {
        self.tools.values().map(|t| t.definition()).collect()
    }

    /// Look up a tool by the name the model called.
    pub fn get(&self, name: &str) -> Option<&Arc<dyn Tool>> {
        self.tools.get(name)
    }

    pub fn len(&self) -> usize {
        self.tools.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }
}

/// The outcome of a [`AlpineClient::run_tools`] loop.
#[derive(Debug, Clone)]
pub struct ToolRun {
    /// The final model response — the one that no longer requests tools.
    pub response: Response,
    /// The full conversation, including the assistant `tool_use` turns and the
    /// `tool_result` messages the loop produced. Feed this back in to continue.
    pub messages: Vec<Message>,
    /// Number of model round-trips made (>= 1).
    pub steps: usize,
    /// Total number of tool invocations executed across all steps.
    pub tool_calls: usize,
}

impl AlpineClient {
    /// Run the model with tools, automatically executing every tool it requests
    /// and feeding the results back, until the model returns a plain answer (no
    /// tool calls) or `max_steps` model round-trips are exhausted.
    ///
    /// `req.tools` is overwritten with the toolbox's definitions, so you don't
    /// need to populate it yourself. Tool errors and calls to unknown tools are
    /// reported back to the model as error `tool_result`s rather than failing
    /// the run — only a provider/transport error or non-convergence aborts.
    pub async fn run_tools(
        &self,
        mut req: Request,
        tools: &Toolbox,
        max_steps: usize,
    ) -> Result<ToolRun, ProviderError> {
        req.tools = tools.definitions();
        let mut messages = req.messages.clone();
        let mut tool_calls = 0usize;

        for step in 1..=max_steps {
            req.messages = messages.clone();
            let resp = self.complete(req.clone()).await?;

            // No tool requests → the model has produced its final answer.
            if resp.tool_calls.is_empty() {
                return Ok(ToolRun {
                    response: resp,
                    messages,
                    steps: step,
                    tool_calls,
                });
            }

            // Record the assistant turn (text + tool_use blocks) so the next
            // request carries the calls these results answer.
            messages.push(resp.assistant_message());

            // Execute each requested tool and append its result.
            for call in &resp.tool_calls {
                tool_calls += 1;
                let (body, is_error) = match tools.get(&call.name) {
                    Some(tool) => match tool.call(call.input.clone()).await {
                        Ok(out) => (out, false),
                        Err(e) => (e.to_string(), true),
                    },
                    None => (format!("error: unknown tool '{}'", call.name), true),
                };
                messages.push(Message::tool_result(&call.id, body, is_error));
            }
        }

        Err(ProviderError::Other(format!(
            "tool loop did not converge within {max_steps} steps"
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provider::Provider;
    use crate::types::{
        ContentBlock, FinishReason, ModelId, Role, StreamResponse, ToolUse, Usage,
    };
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    // A tool that echoes its `text` argument and counts how many times it ran.
    struct Echo {
        calls: Arc<AtomicUsize>,
    }

    #[async_trait]
    impl Tool for Echo {
        fn definition(&self) -> ToolDefinition {
            ToolDefinition::new(
                "echo",
                "Echo text",
                serde_json::json!({"type": "object"}),
            )
        }

        async fn call(&self, input: serde_json::Value) -> Result<String, ToolError> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            Ok(input["text"].as_str().unwrap_or("").to_string())
        }
    }

    // A tool that always fails.
    struct Boom;

    #[async_trait]
    impl Tool for Boom {
        fn definition(&self) -> ToolDefinition {
            ToolDefinition::new("boom", "Always fails", serde_json::json!({"type": "object"}))
        }

        async fn call(&self, _input: serde_json::Value) -> Result<String, ToolError> {
            Err("kaboom".into())
        }
    }

    fn tool_use_response(name: &str, id: &str) -> Response {
        Response {
            content: String::new(),
            tool_calls: vec![ToolUse {
                id: id.into(),
                name: name.into(),
                input: serde_json::json!({"text": "hi"}),
            }],
            usage: Usage::default(),
            model: ModelId::new("scripted"),
            finish_reason: FinishReason::ToolUse,
            latency: Duration::ZERO,
            raw: serde_json::Value::Null,
        }
    }

    fn final_response(text: &str) -> Response {
        Response {
            content: text.into(),
            tool_calls: vec![],
            usage: Usage::default(),
            model: ModelId::new("scripted"),
            finish_reason: FinishReason::Stop,
            latency: Duration::ZERO,
            raw: serde_json::Value::Null,
        }
    }

    /// Provider that requests `tool_name` once, then — once it sees any
    /// `tool_result` in the conversation — returns a final answer. Optionally
    /// keeps requesting forever (to exercise the non-convergence guard).
    struct ScriptedProvider {
        model: ModelId,
        tool_name: String,
        loop_forever: bool,
    }

    impl ScriptedProvider {
        fn new(tool_name: &str) -> Self {
            Self {
                model: ModelId::new("scripted"),
                tool_name: tool_name.into(),
                loop_forever: false,
            }
        }
    }

    fn has_tool_result(req: &Request) -> bool {
        req.messages
            .iter()
            .flat_map(|m| m.content.iter())
            .any(|b| matches!(b, ContentBlock::ToolResult { .. }))
    }

    #[async_trait]
    impl Provider for ScriptedProvider {
        async fn complete(&self, req: &Request) -> Result<Response, ProviderError> {
            if self.loop_forever {
                return Ok(tool_use_response(&self.tool_name, "tu"));
            }
            if has_tool_result(req) {
                Ok(final_response("done"))
            } else {
                Ok(tool_use_response(&self.tool_name, "tu_1"))
            }
        }

        async fn stream(&self, _req: &Request) -> Result<StreamResponse<'_>, ProviderError> {
            Ok(Box::pin(futures::stream::empty()))
        }

        fn model_id(&self) -> &ModelId {
            &self.model
        }
    }

    #[test]
    fn toolbox_registers_and_lists() {
        let calls = Arc::new(AtomicUsize::new(0));
        let tb = Toolbox::new().with(Echo { calls }).with(Boom);
        assert_eq!(tb.len(), 2);
        assert!(tb.get("echo").is_some());
        assert!(tb.get("boom").is_some());
        assert!(tb.get("missing").is_none());
        let names: Vec<_> = tb.definitions().into_iter().map(|d| d.name).collect();
        assert!(names.contains(&"echo".to_string()));
        assert!(names.contains(&"boom".to_string()));
    }

    #[test]
    fn toolbox_same_name_replaces() {
        let tb = Toolbox::new().with(Boom).with(Boom);
        assert_eq!(tb.len(), 1);
    }

    #[tokio::test]
    async fn run_tools_executes_and_feeds_result_back() {
        let calls = Arc::new(AtomicUsize::new(0));
        let tools = Toolbox::new().with(Echo {
            calls: calls.clone(),
        });
        let client = AlpineClient::new(ScriptedProvider::new("echo"));
        let req = Request {
            messages: vec![Message::user("use echo")],
            ..Default::default()
        };

        let run = client.run_tools(req, &tools, 8).await.unwrap();

        assert_eq!(run.response.content, "done");
        assert_eq!(run.steps, 2); // request tool, then final answer
        assert_eq!(run.tool_calls, 1);
        assert_eq!(calls.load(Ordering::SeqCst), 1);

        // Conversation: user, assistant(tool_use), user(tool_result).
        assert_eq!(run.messages.len(), 3);
        assert_eq!(run.messages[1].role, Role::Assistant);
        assert!(run.messages[1]
            .content
            .iter()
            .any(|b| matches!(b, ContentBlock::ToolUse { .. })));
        assert!(matches!(
            run.messages[2].content[0],
            ContentBlock::ToolResult { is_error: false, .. }
        ));
    }

    #[tokio::test]
    async fn run_tools_reports_tool_error_to_model() {
        let tools = Toolbox::new().with(Boom);
        let client = AlpineClient::new(ScriptedProvider::new("boom"));
        let req = Request {
            messages: vec![Message::user("use boom")],
            ..Default::default()
        };

        let run = client.run_tools(req, &tools, 8).await.unwrap();

        // Loop still converges; the failure surfaces as an error tool_result.
        assert_eq!(run.response.content, "done");
        let result_block = &run.messages[2].content[0];
        match result_block {
            ContentBlock::ToolResult {
                content, is_error, ..
            } => {
                assert!(*is_error);
                assert!(content.contains("kaboom"));
            }
            other => panic!("expected error ToolResult, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn run_tools_reports_unknown_tool() {
        // Model asks for "ghost" but the toolbox only has echo.
        let calls = Arc::new(AtomicUsize::new(0));
        let tools = Toolbox::new().with(Echo { calls });
        let client = AlpineClient::new(ScriptedProvider::new("ghost"));
        let req = Request {
            messages: vec![Message::user("use ghost")],
            ..Default::default()
        };

        let run = client.run_tools(req, &tools, 8).await.unwrap();
        match &run.messages[2].content[0] {
            ContentBlock::ToolResult {
                content, is_error, ..
            } => {
                assert!(*is_error);
                assert!(content.contains("unknown tool 'ghost'"));
            }
            other => panic!("expected error ToolResult, got {other:?}"),
        }
    }

    // A provider that always returns a plain final answer, never a tool call.
    struct Plain(ModelId);

    #[async_trait]
    impl Provider for Plain {
        async fn complete(&self, _req: &Request) -> Result<Response, ProviderError> {
            Ok(final_response("hi"))
        }
        async fn stream(&self, _req: &Request) -> Result<StreamResponse<'_>, ProviderError> {
            Ok(Box::pin(futures::stream::empty()))
        }
        fn model_id(&self) -> &ModelId {
            &self.0
        }
    }

    #[tokio::test]
    async fn run_tools_stops_immediately_without_tool_calls() {
        let client = AlpineClient::new(Plain(ModelId::new("plain")));
        let run = client
            .run_tools(
                Request {
                    messages: vec![Message::user("hello")],
                    ..Default::default()
                },
                &Toolbox::new(),
                4,
            )
            .await
            .unwrap();
        assert_eq!(run.response.content, "hi");
        assert_eq!(run.steps, 1);
        assert_eq!(run.tool_calls, 0);
        assert_eq!(run.messages.len(), 1);
    }

    #[tokio::test]
    async fn run_tools_errors_when_not_converging() {
        let calls = Arc::new(AtomicUsize::new(0));
        let tools = Toolbox::new().with(Echo { calls });
        let client = AlpineClient::new(ScriptedProvider {
            model: ModelId::new("scripted"),
            tool_name: "echo".into(),
            loop_forever: true,
        });
        let err = client
            .run_tools(
                Request {
                    messages: vec![Message::user("spin")],
                    ..Default::default()
                },
                &tools,
                3,
            )
            .await
            .unwrap_err();
        assert!(err.to_string().contains("did not converge"));
    }
}
