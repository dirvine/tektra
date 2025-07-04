use super::*;
use futures::{Stream, StreamExt};
use std::pin::Pin;
use tokio::sync::mpsc;

/// Streaming response handler for real-time token generation
pub struct StreamingHandler {
    sender: mpsc::Sender<Result<Token>>,
    receiver: Option<mpsc::Receiver<Result<Token>>>,
    buffer_size: usize,
}

impl StreamingHandler {
    pub fn new(buffer_size: usize) -> Self {
        let (sender, receiver) = mpsc::channel(buffer_size);
        
        Self {
            sender,
            receiver: Some(receiver),
            buffer_size,
        }
    }
    
    /// Send a token to the stream
    pub async fn send_token(&self, token: Token) -> Result<()> {
        self.sender.send(Ok(token)).await
            .map_err(|_| anyhow::anyhow!("Failed to send token to stream"))?;
        Ok(())
    }
    
    /// Send an error to the stream
    pub async fn send_error(&self, error: anyhow::Error) -> Result<()> {
        self.sender.send(Err(error)).await
            .map_err(|_| anyhow::anyhow!("Failed to send error to stream"))?;
        Ok(())
    }
    
    /// Get the receiving end of the stream
    pub fn into_stream(mut self) -> Pin<Box<dyn Stream<Item = Result<Token>> + Send>> {
        let receiver = self.receiver.take()
            .expect("Stream receiver already taken");
        
        Box::pin(tokio_stream::wrappers::ReceiverStream::new(receiver))
    }
    
    /// Get sender handle for external use
    pub fn get_sender(&self) -> mpsc::Sender<Result<Token>> {
        self.sender.clone()
    }
}

/// Stream processor for handling different types of streaming responses
pub struct StreamProcessor {
    buffer: String,
    token_count: usize,
    start_time: std::time::Instant,
}

impl StreamProcessor {
    pub fn new() -> Self {
        Self {
            buffer: String::new(),
            token_count: 0,
            start_time: std::time::Instant::now(),
        }
    }
    
    /// Process a raw text chunk and create appropriate tokens
    pub fn process_chunk(&mut self, chunk: &str) -> Vec<Token> {
        self.buffer.push_str(chunk);
        self.token_count += 1;
        
        // For now, treat each chunk as a single token
        // In a more sophisticated implementation, this could handle
        // word boundaries, sentence completion, etc.
        vec![Token {
            text: chunk.to_string(),
            logprob: None,
            special: false,
        }]
    }
    
    /// Finalize the stream and return summary statistics
    pub fn finalize(&self) -> StreamSummary {
        StreamSummary {
            total_tokens: self.token_count,
            total_text: self.buffer.clone(),
            processing_time_ms: self.start_time.elapsed().as_millis() as u64,
            tokens_per_second: if self.start_time.elapsed().as_secs() > 0 {
                self.token_count as f64 / self.start_time.elapsed().as_secs_f64()
            } else {
                0.0
            },
        }
    }
    
    /// Reset the processor for reuse
    pub fn reset(&mut self) {
        self.buffer.clear();
        self.token_count = 0;
        self.start_time = std::time::Instant::now();
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamSummary {
    pub total_tokens: usize,
    pub total_text: String,
    pub processing_time_ms: u64,
    pub tokens_per_second: f64,
}

/// Utility for merging multiple streams
pub struct StreamMerger {
    streams: Vec<Pin<Box<dyn Stream<Item = Result<Token>> + Send>>>,
}

impl StreamMerger {
    pub fn new() -> Self {
        Self {
            streams: Vec::new(),
        }
    }
    
    pub fn add_stream(&mut self, stream: Pin<Box<dyn Stream<Item = Result<Token>> + Send>>) {
        self.streams.push(stream);
    }
    
    pub fn into_merged_stream(self) -> Pin<Box<dyn Stream<Item = Result<Token>> + Send>> {
        if self.streams.is_empty() {
            return Box::pin(futures::stream::empty());
        }
        
        if self.streams.len() == 1 {
            return self.streams.into_iter().next().unwrap();
        }
        
        // For multiple streams, we could implement various merging strategies
        // For now, just concatenate them sequentially
        Box::pin(futures::stream::iter(self.streams).flatten())
    }
}

/// Rate limiter for streaming to prevent overwhelming clients
pub struct StreamRateLimiter {
    max_tokens_per_second: usize,
    token_bucket: usize,
    last_refill: std::time::Instant,
}

impl StreamRateLimiter {
    pub fn new(max_tokens_per_second: usize) -> Self {
        Self {
            max_tokens_per_second,
            token_bucket: max_tokens_per_second,
            last_refill: std::time::Instant::now(),
        }
    }
    
    /// Check if a token can be sent now, and wait if necessary
    pub async fn check_rate_limit(&mut self) -> bool {
        self.refill_bucket();
        
        if self.token_bucket > 0 {
            self.token_bucket -= 1;
            true
        } else {
            // Calculate how long to wait
            let tokens_needed = 1;
            let wait_time = std::time::Duration::from_millis(
                (tokens_needed * 1000) as u64 / self.max_tokens_per_second as u64
            );
            
            tokio::time::sleep(wait_time).await;
            self.refill_bucket();
            
            if self.token_bucket > 0 {
                self.token_bucket -= 1;
                true
            } else {
                false
            }
        }
    }
    
    fn refill_bucket(&mut self) {
        let now = std::time::Instant::now();
        let elapsed = now.duration_since(self.last_refill);
        
        if elapsed >= std::time::Duration::from_secs(1) {
            self.token_bucket = self.max_tokens_per_second;
            self.last_refill = now;
        }
    }
}

/// Transform stream tokens with various processing options
pub struct StreamTransformer {
    options: TransformOptions,
}

#[derive(Debug, Clone)]
pub struct TransformOptions {
    pub filter_special_tokens: bool,
    pub word_boundary_detection: bool,
    pub sentence_boundary_detection: bool,
    pub format_markdown: bool,
    pub real_time_processing: bool,
}

impl Default for TransformOptions {
    fn default() -> Self {
        Self {
            filter_special_tokens: true,
            word_boundary_detection: true,
            sentence_boundary_detection: false,
            format_markdown: false,
            real_time_processing: true,
        }
    }
}

impl StreamTransformer {
    pub fn new(options: TransformOptions) -> Self {
        Self { options }
    }
    
    /// Transform a stream of tokens according to the specified options
    pub fn transform_stream(
        &self,
        stream: Pin<Box<dyn Stream<Item = Result<Token>> + Send>>,
    ) -> Pin<Box<dyn Stream<Item = Result<Token>> + Send + '_>> {
        let options = self.options.clone();
        
        let transformed = stream.map(move |token_result| {
            match token_result {
                Ok(token) => {
                    let mut transformed_token = token;
                    
                    // Filter special tokens if requested
                    if options.filter_special_tokens && transformed_token.special {
                        return Ok(Token {
                            text: String::new(),
                            logprob: transformed_token.logprob,
                            special: true,
                        });
                    }
                    
                    // Word boundary detection
                    if options.word_boundary_detection {
                        // Add logic for word boundary processing
                        // This could involve buffering partial words
                    }
                    
                    // Sentence boundary detection
                    if options.sentence_boundary_detection {
                        // Add logic for sentence boundary processing
                    }
                    
                    // Markdown formatting
                    if options.format_markdown {
                        transformed_token.text = self.format_as_markdown(&transformed_token.text);
                    }
                    
                    Ok(transformed_token)
                }
                Err(e) => Err(e),
            }
        });
        
        Box::pin(transformed)
    }
    
    fn format_as_markdown(&self, text: &str) -> String {
        // Basic markdown formatting
        // This could be expanded with more sophisticated formatting rules
        text.to_string()
    }
}

/// Buffer for collecting streaming tokens into complete responses
pub struct StreamBuffer {
    tokens: Vec<Token>,
    complete_text: String,
    max_buffer_size: usize,
}

impl StreamBuffer {
    pub fn new(max_buffer_size: usize) -> Self {
        Self {
            tokens: Vec::new(),
            complete_text: String::new(),
            max_buffer_size,
        }
    }
    
    /// Add a token to the buffer
    pub fn add_token(&mut self, token: Token) -> Result<()> {
        if self.tokens.len() >= self.max_buffer_size {
            return Err(anyhow::anyhow!("Stream buffer overflow"));
        }
        
        if !token.special {
            self.complete_text.push_str(&token.text);
        }
        
        self.tokens.push(token);
        Ok(())
    }
    
    /// Get the complete text so far
    pub fn get_text(&self) -> &str {
        &self.complete_text
    }
    
    /// Get all tokens
    pub fn get_tokens(&self) -> &[Token] {
        &self.tokens
    }
    
    /// Check if buffer is complete (ended with stop token or reached limit)
    pub fn is_complete(&self) -> bool {
        if let Some(last_token) = self.tokens.last() {
            last_token.special || self.tokens.len() >= self.max_buffer_size
        } else {
            false
        }
    }
    
    /// Convert to final model response
    pub fn into_response(self, finish_reason: FinishReason) -> ModelResponse {
        let total_tokens = self.tokens.len();
        let inference_time = 0; // Would need to track actual time
        
        ModelResponse {
            text: self.complete_text,
            tokens: self.tokens,
            finish_reason,
            usage: UsageStats {
                prompt_tokens: 0, // Would need to be tracked separately
                completion_tokens: total_tokens,
                total_tokens,
                inference_time_ms: inference_time,
            },
            audio: None,
            metadata: crate::inference::ResponseMetadata::default(),
        }
    }
    
    /// Clear the buffer for reuse
    pub fn clear(&mut self) {
        self.tokens.clear();
        self.complete_text.clear();
    }
}