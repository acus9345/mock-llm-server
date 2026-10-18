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

export interface ReplyObject {
  /** The assistant's text. */
  text?: string;
  /** Shorthand for `text: JSON.stringify(json)`, for structured-output tests. */
  json?: unknown;
  /** Tool calls, after any text. */
  toolCalls?: ToolCall[];
  /** Return an HTTP error instead of a completion. */
  error?: ReplyError;
  /** Why the model stopped. Default: `"tool_calls"` when there are tool calls, else `"stop"`. Mapped to each provider's wording. */
  stopReason?: "stop" | "length" | "tool_calls" | "content_filter";
  /** Token counts. Default: estimated as characters / 4. */
  usage?: { inputTokens?: number; outputTokens?: number };
  /** Model name in the response. Default: the requested model. */
  model?: string;
  /** Wait this long before responding. */
  delayMs?: number;
  /** Characters per streamed chunk. Default: one word per chunk. */
  chunkSize?: number;
  /** Wait this long between streamed chunks. */
  chunkDelayMs?: number;
}

/** A recorded request, also passed to matchers and reply functions. */
export interface RecordedRequest {
  /** 1-based count of model requests the server has handled. */
  index: number;
  provider: "openai" | "anthropic";
  api: "chat.completions" | "responses" | "messages" | "embeddings" | "count_tokens";
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
  toolCalls: { id: string; name: string; arguments: string }[];
  error: Required<Pick<ReplyError, "status" | "retryAfterMs">> & Pick<ReplyError, "message" | "type"> | undefined;
  stopReason: NonNullable<ReplyObject["stopReason"]>;
  inputTokens: number;
  outputTokens: number;
  model: string;
  delayMs: number;
  chunkSize: number | undefined;
  chunkDelayMs: number;
}
