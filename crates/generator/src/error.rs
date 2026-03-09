use thiserror::Error;

#[derive(Debug, Error)]
pub enum GeneratorError {
    #[error("card generation failed: {message}")]
    Generation {
        message: String,
        model: Option<String>,
    },

    #[error("card validation failed: {message}")]
    Validation { message: String },

    #[error("card enhancement failed: {message}")]
    Enhancement {
        message: String,
        model: Option<String>,
    },

    #[error("APF format error: {0}")]
    Apf(String),

    #[error("HTML conversion error: {0}")]
    HtmlConversion(String),

    #[error("LLM error: {0}")]
    Llm(#[from] llm::LlmError),
}
