/** A tool call the mock model makes. */
export interface ToolCall {
  name: string;
  /** Object (sent as JSON) or a raw JSON string. Default `{}`. */
  arguments?: Record<string, unknown> | string;
  /** Default: a generated id like `call_mock_1` / `toolu_mock_1`. */
  id?: string;
}

/** An HTTP error the mock returns instead of a completion. */
export interface ReplyError {
  /** HTTP status. Default 500. */
  status?: number;
  message?: string;
  /** Provider error type, e.g. `"rate_limit_error"`. Default: derived from `status`. */
  type?: string;
  /** Sent as `retry-after-ms` / `retry-after` headers. Default 0, so SDK retries happen at once. */
  retryAfterMs?: number;
}

/** Make a streamed response fail partway through. */
export interface StreamError {
  /** How many content chunks (text, reasoning or tool-call pieces) to send before failing. Default 0. */
  afterChunks?: number;
  /** Error message. Default: `"Mock stream error"`. */
  message?: string;
  /** Provider error type, e.g. `"overloaded_error"`. Default: derived from `status`. */
  type?: string;
  /** Status used to pick the error type, and the HTTP status when the request isn't streamed. Default 500. */
  status?: number;
  /** Drop the connection instead of sending an error event, like a network failure. */
  disconnect?: boolean;
}

export interface ReplyObject {
  /** The assistant's text. */
  text?: string;
  /**
   * Reasoning shown before the answer: an Anthropic `thinking` block, an OpenAI Responses `reasoning`
   * item with a summary, a Gemini thought part, and `reasoning_content` in Chat Completions.
   */
  reasoning?: string;
  /** Shorthand for `text: JSON.stringify(json)`, for structured-output tests. */
  json?: unknown;
  /** Tool calls, after any text. */
  toolCalls?: ToolCall[];
  /** Return an HTTP error instead of a completion. */
  error?: ReplyError;
  /** Why the model stopped. Default: `"tool_calls"` when there are tool calls, else `"stop"`. Mapped to each provider's wording. */
  stopReason?: "stop" | "length" | "tool_calls" | "content_filter";
  /**
   * Token counts, sent as given in each provider's fields. Default: estimated as characters / 4.
   * `cachedInputTokens` and `cacheWriteTokens` default to 0, `reasoningTokens` to the estimate for `reasoning`.
   */
  usage?: {
    inputTokens?: number;
    outputTokens?: number;
    cachedInputTokens?: number;
    cacheWriteTokens?: number;
    reasoningTokens?: number;
  };
  /** Model name in the response. Default: the requested model. */
  model?: string;
  /** Wait this long before responding. */
  delayMs?: number;
  /** Characters per streamed chunk. Default: one word per chunk. */
  chunkSize?: number;
  /** Wait this long between streamed chunks. */
  chunkDelayMs?: number;
  /** Fail partway through the stream with an error event, or by dropping the connection. */
  streamError?: StreamError;
}

/** A recorded request, also passed to matchers and reply functions. */
export interface RecordedRequest {
  /** 1-based count of model requests the server has handled. */
  index: number;
  provider: "openai" | "anthropic" | "gemini";
  api: "chat.completions" | "responses" | "messages" | "generate_content" | "embeddings" | "count_tokens";
  method: string;
  path: string;
  headers: Record<string, string>;
  /** Parsed JSON body. */
  body: Record<string, unknown>;
  model: string;
  stream: boolean;
  /** System prompt or instructions, as text. */
  system: string;
  /** Conversation flattened to text. Tool results are `role: "tool"`. */
  messages: { role: string; content: string }[];
  /** Text of the last user message ("" if the last turn was only tool results). */
  lastUserMessage: string;
  /** Names of the tools offered in the request. */
  tools: string[];
}

export type Reply = string | ReplyObject | ((request: RecordedRequest) => Reply | Promise<Reply>);

/** A substring of the last user message, a RegExp tested against it, or a predicate on the request. */
export type Matcher = string | RegExp | ((request: RecordedRequest) => boolean);

/** A fully resolved reply, ready to render for any provider. */
export interface ResolvedReply {
  text: string;
  reasoning: string;
  /** `idGiven` is false when the id was generated. */
  toolCalls: { id: string; name: string; arguments: string; idGiven: boolean }[];
  error: Required<Pick<ReplyError, "status" | "retryAfterMs">> & Pick<ReplyError, "message" | "type"> | undefined;
  stopReason: NonNullable<ReplyObject["stopReason"]>;
  inputTokens: number;
  outputTokens: number;
  cachedInputTokens: number;
  cacheWriteTokens: number;
  reasoningTokens: number;
  model: string;
  delayMs: number;
  chunkSize: number | undefined;
  chunkDelayMs: number;
  streamError: Required<Pick<StreamError, "afterChunks" | "message" | "status" | "disconnect">> & Pick<StreamError, "type"> | undefined;
}
