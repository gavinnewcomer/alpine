use thiserror::Error;

#[derive(Debug, Error)]
pub enum ProviderError {
    #[error("HTTP error: {0}")]
    Http(#[from] reqwest::Error),

    #[error("JSON serialization error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("API error ({status}): {message}")]
    Api { status: u16, message: String },

    #[error("Stream error: {0}")]
    Stream(String),

    #[error("Unsupported operation: {0}")]
    Unsupported(String),

    #[error("{0}")]
    Other(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn api_error_display() {
        let e = ProviderError::Api {
            status: 429,
            message: "rate limit".into(),
        };
        let s = e.to_string();
        assert!(s.contains("429"));
        assert!(s.contains("rate limit"));
    }

    #[test]
    fn stream_error_display() {
        let e = ProviderError::Stream("broken pipe".into());
        assert!(e.to_string().contains("broken pipe"));
    }

    #[test]
    fn unsupported_display() {
        let e = ProviderError::Unsupported("embed".into());
        assert!(e.to_string().contains("embed"));
    }

    #[test]
    fn other_display() {
        let e = ProviderError::Other("oops".into());
        assert_eq!(e.to_string(), "oops");
    }

    #[test]
    fn json_error_from_serde() {
        let err = serde_json::from_str::<serde_json::Value>("not json").unwrap_err();
        let pe = ProviderError::from(err);
        assert!(pe.to_string().contains("JSON serialization error"));
    }

    #[test]
    fn http_error_from_reqwest() {
        // Build a reqwest error by sending to an invalid URL
        let rt = tokio::runtime::Runtime::new().unwrap();
        let err = rt.block_on(async { reqwest::get("http://[::0]:0/invalid").await.unwrap_err() });
        let pe = ProviderError::from(err);
        assert!(pe.to_string().contains("HTTP error"));
    }

    #[test]
    fn error_is_std_error() {
        let e: Box<dyn std::error::Error> = Box::new(ProviderError::Other("x".into()));
        assert!(e.source().is_none());

        let json_err = serde_json::from_str::<serde_json::Value>("bad").unwrap_err();
        let e: Box<dyn std::error::Error> = Box::new(ProviderError::Json(json_err));
        assert!(e.source().is_some());
    }
}
