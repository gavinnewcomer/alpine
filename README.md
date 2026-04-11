# Alpine

Alpine provides a lightweight, low-bulk interface for all major LLM providers. It exposes composable types for responses, which allows for simple chat sessions or can be plugged into larger projects.

One `Request` type, one `Response` type, swap providers with a single line.

## Features

- **Unified types** -- `Request`, `Response`, `Message`, `Usage` work the same across every provider
- **Feature-gated providers** -- only compile what you use
- **Streaming** -- first-class `StreamChunk` support via async streams
- **Embeddings** -- where the provider supports it (Ollama)
- **Middleware** -- pluggable request/response pipeline on `AlpineClient`
- **Latency tracking** -- every `Response` includes wall-clock `latency`

## Installation

Add Alpine to your `Cargo.toml` with the providers you need:

```toml
[dependencies]
alpine = { path = "../alpine", features = ["ollama"] }
# or
alpine = { path = "../alpine", features = ["anthropic"] }
# or both
alpine = { path = "../alpine", features = ["ollama", "anthropic"] }
```

## Quick start

### Ollama

Requires [Ollama](https://ollama.com) running locally on the default port.

```rust
use alpine::{AlpineClient, Message, Request};
use alpine::providers::ollama::OllamaProvider;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = AlpineClient::new(OllamaProvider::new("llama3.2"));

    let res = client.complete(Request {
        messages: vec![Message::user("Explain async/await in Rust in two sentences.")],
        ..Default::default()
    }).await?;

    println!("{res}");
    // [llama3.2] (1.24s)
    // Async/await in Rust allows you to write asynchronous code that looks synchronous...
    // tokens: 28 in / 64 out | finish: Stop

    Ok(())
}
```

### Anthropic

Requires an API key from [Anthropic](https://console.anthropic.com).

```rust
use alpine::{AlpineClient, Message, Request};
use alpine::providers::anthropic::AnthropicProvider;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let api_key = std::env::var("ANTHROPIC_API_KEY")?;
    let client = AlpineClient::new(
        AnthropicProvider::new(api_key, "claude-sonnet-4-20250514"),
    );

    let res = client.complete(Request {
        messages: vec![Message::user("What is the capital of France?")],
        ..Default::default()
    }).await?;

    println!("{res}");

    Ok(())
}
```

### Streaming

Both providers support token-by-token streaming:

```rust
use futures::StreamExt;
use alpine::{AlpineClient, Message, Request, StreamChunk};
use alpine::providers::ollama::OllamaProvider;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = AlpineClient::new(OllamaProvider::new("llama3.2"));

    let req = Request {
        messages: vec![Message::user("Count to five slowly.")],
        ..Default::default()
    };

    let mut stream = client.stream(&req).await?;

    while let Some(chunk) = stream.next().await {
        match chunk {
            StreamChunk::Delta(text) => print!("{text}"),
            StreamChunk::Done { usage } => {
                println!("\n-- done, usage: {usage:?}");
            }
            StreamChunk::Error(e) => eprintln!("\nerror: {e}"),
        }
    }

    Ok(())
}
```

### Embeddings (Ollama)

```rust
use alpine::providers::ollama::OllamaProvider;
use alpine::provider::Provider;
use alpine::{EmbedRequest, ModelId};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let provider = OllamaProvider::new("nomic-embed-text");

    let embedding = provider.embed(&EmbedRequest {
        model: ModelId::new("nomic-embed-text"),
        input: vec!["Hello, world!".into()],
    }).await?;

    println!("dimensions: {}", embedding.vectors[0].len());

    Ok(())
}
```

### Swapping providers

The whole point -- same request works with any provider:

```rust
use alpine::{AlpineClient, Message, Request};
use alpine::providers::ollama::OllamaProvider;
use alpine::providers::anthropic::AnthropicProvider;

fn build_client(use_local: bool) -> AlpineClient {
    if use_local {
        AlpineClient::new(OllamaProvider::new("llama3.2"))
    } else {
        let key = std::env::var("ANTHROPIC_API_KEY").unwrap();
        AlpineClient::new(AnthropicProvider::new(key, "claude-sonnet-4-20250514"))
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = build_client(cfg!(debug_assertions));

    let res = client.complete(Request {
        messages: vec![Message::user("hello")],
        ..Default::default()
    }).await?;

    // Same Response type regardless of provider
    println!("model: {}", res.model);
    println!("latency: {:.0?}", res.latency);
    println!("tokens: {} in, {} out", res.usage.input_tokens, res.usage.output_tokens);
    println!("{}", res.content);

    Ok(())
}
```

## Response

Every `complete()` call returns a `Response` with:

| Field | Type | Description |
|-------|------|-------------|
| `content` | `String` | The generated text |
| `usage` | `Usage` | `input_tokens` and `output_tokens` |
| `model` | `ModelId` | The model that served the request |
| `finish_reason` | `FinishReason` | `Stop`, `MaxTokens`, `ContentFilter`, or `Other(String)` |
| `latency` | `Duration` | Wall-clock time for the provider round-trip |
| `raw` | `serde_json::Value` | The unmodified provider response (escape hatch) |

## License

MIT
