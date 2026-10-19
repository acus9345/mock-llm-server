# Changelog

All notable changes to this project are documented here. This project follows [Semantic Versioning](https://semver.org/).

## [0.2.0] - 2026-10-19

### Added

- Gemini's native API: `generateContent`, `streamGenerateContent`, `countTokens`, `embedContent`, `batchEmbedContents` and the model list, tested with `@google/genai`. `mock.geminiBaseURL` is the base URL to pass to the SDK.
- `reasoning` replies: an Anthropic `thinking` block with a signature, an OpenAI Responses `reasoning` item with a summary, a Gemini thought part, and `reasoning_content` in Chat Completions. All of them stream.
- `streamError` replies that end a stream with the provider's error event after a number of chunks, or drop the connection.
- `usage.cachedInputTokens`, `usage.cacheWriteTokens` and `usage.reasoningTokens`, sent in each provider's usage fields.
- `mock.waitForRequests(count, { timeoutMs })` and the `GET /__mock/wait` admin route, for agents that run in the background.

## [0.1.0] - 2026-10-18

### Added

- `startMockLLM()` starts a fake API server with OpenAI Chat Completions, Responses and Embeddings, and Anthropic Messages and token counting, plus model lists for both.
- Streaming over SSE in each API's event format, with text in word-sized chunks and tool-call arguments in pieces. Works with the official SDKs' streaming helpers.
- Replies as text, JSON, tool calls, provider-format errors, stop reasons, token usage, delays and chunk settings.
- Scripting with a reply queue, matching rules (`times` supported), reply functions and an `echo` fallback. Unscripted requests get an error that explains what is missing.
- Recorded requests with system prompt, flattened messages, last user message and tool names; `assertDone()`, `reset()` and `await using` support.
- Error replies send `retry-after-ms: 0` so SDK retries happen at once.
- `mock-llm-server` CLI with a JSON script file and HTTP admin routes (`/__mock/replies`, `/__mock/rules`, `/__mock/requests`, `/__mock/reset`) for tests in other languages.

[0.2.0]: https://github.com/acus9345/mock-llm-server/releases/tag/v0.2.0
[0.1.0]: https://github.com/acus9345/mock-llm-server/releases/tag/v0.1.0
