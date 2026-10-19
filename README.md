# mock-llm-server

A fake OpenAI-, Anthropic- and Gemini-compatible API that returns scripted replies, tool calls, reasoning, errors and streams, so you can unit test agents without API keys. Zero dependencies.

[![npm](https://img.shields.io/npm/v/mock-llm-server)](https://www.npmjs.com/package/mock-llm-server)
[![CI](https://github.com/acus9345/mock-llm-server/actions/workflows/ci.yml/badge.svg)](https://github.com/acus9345/mock-llm-server/actions/workflows/ci.yml)
[![license](https://img.shields.io/npm/l/mock-llm-server)](LICENSE)

- **Real wire formats:** OpenAI Chat Completions, Responses and Embeddings, Anthropic Messages and token counting, Gemini `generateContent`, streaming, token counting and embeddings. Tested against the official `openai`, `@anthropic-ai/sdk` and `@google/genai` packages, including their streaming helpers.
- **Streaming and tool calls:** text arrives word by word over SSE, and tool-call arguments arrive in pieces, just like the real APIs.
- **Reasoning:** Anthropic thinking blocks, OpenAI reasoning summaries, Gemini thought parts and DeepSeek-style `reasoning_content`.
- **Failures:** HTTP errors in each provider's format, plus streams that break partway or drop the connection.
- **Script it your way:** a queue of replies, rules that match the user's message, or functions that see the whole request.
- **Inspect what was sent:** every request is recorded with its system prompt, flattened messages and tool names.
- **Any language:** run it as a CLI and script it over HTTP from Python, Go or anything else.

## Install

```bash
npm install --save-dev mock-llm-server
```

## Quick start

```js
import OpenAI from "openai";
import { startMockLLM } from "mock-llm-server";

const mock = await startMockLLM();
const openai = new OpenAI({ baseURL: mock.openaiBaseURL, apiKey: "test" });

mock.reply("Paris is the capital of France.");

const res = await openai.chat.completions.create({
  model: "gpt-5",
  messages: [{ role: "user", content: "What is the capital of France?" }],
});
console.log(res.choices[0].message.content); // Paris is the capital of France.
console.log(mock.lastRequest.lastUserMessage); // What is the capital of France?

await mock.close();
```

Anthropic works the same way:

```js
import Anthropic from "@anthropic-ai/sdk";

const anthropic = new Anthropic({ baseURL: mock.anthropicBaseURL, apiKey: "test" });
mock.reply("Bonjour !");
const msg = await anthropic.messages.create({ model: "claude-sonnet-5-5", max_tokens: 100, messages });
```

So does Gemini:

```js
import { GoogleGenAI } from "@google/genai";

const ai = new GoogleGenAI({ apiKey: "test", httpOptions: { baseUrl: mock.geminiBaseURL } });
mock.reply("Hallo!");
const res = await ai.models.generateContent({ model: "gemini-2.5-flash", contents: "Say hi in German" });
console.log(res.text); // Hallo!
```

## Testing an agent loop

```js
import { test } from "node:test";
import assert from "node:assert/strict";

test("the agent calls the weather tool, then answers", async () => {
  await using mock = await startMockLLM();
  mock.reply(
    { toolCalls: [{ name: "get_weather", arguments: { city: "Oslo" } }] },
    "It's 4 °C and raining in Oslo.",
  );

  const answer = await runMyAgent("What's the weather in Oslo?", { baseURL: mock.openaiBaseURL });

  assert.equal(answer, "It's 4 °C and raining in Oslo.");
  assert.equal(mock.requests.length, 2);
  // The second request carried the tool result back to the model.
  assert.equal(mock.requests[1].messages.at(-1).role, "tool");
  mock.assertDone(); // every queued reply was used
});
```

## Replies

A reply is a string, an object, or a function that returns either.

```js
mock.reply("Plain text");
mock.reply({ text: "Text and a tool call", toolCalls: [{ name: "search", arguments: { q: "llm" } }] });
mock.reply({ json: { name: "Ada", age: 36 } }); // text is JSON.stringify(json), for structured output
mock.reply({ error: { status: 429, message: "Rate limited" } });
mock.reply({ text: "Cut off mid-sent", stopReason: "length" });
mock.reply({ text: "Slow stream", chunkDelayMs: 50, delayMs: 200 });
mock.reply({ reasoning: "The user wants a haiku.", text: "Autumn moonlight..." });
mock.reply({ text: "This stream breaks", streamError: { afterChunks: 2, message: "Upstream died" } });
mock.reply((req) => `You sent ${req.messages.length} messages to ${req.model}`);
```

| Field | What it does |
| --- | --- |
| `text` | The assistant's text. |
| `reasoning` | Reasoning before the answer: an Anthropic `thinking` block (with a signature), an OpenAI Responses `reasoning` item with a summary, a Gemini part with `thought: true`, and `reasoning_content` in Chat Completions (what DeepSeek and vLLM send). |
| `json` | Shorthand for `text: JSON.stringify(json)`. |
| `toolCalls` | `[{ name, arguments?, id? }]`. `arguments` is an object or a JSON string. Ids are generated if missing. |
| `error` | `{ status, message?, type?, retryAfterMs? }`. Sent in the provider's error format, so the SDK throws its usual typed error (`RateLimitError` and so on). |
| `stopReason` | `"stop"`, `"length"`, `"tool_calls"` or `"content_filter"`, mapped to each API's wording (`end_turn`, `max_tokens`, `incomplete`...). Inferred if missing. |
| `usage` | `{ inputTokens, outputTokens, cachedInputTokens, cacheWriteTokens, reasoningTokens }`, sent as given in each provider's fields (`cached_tokens`, `cache_read_input_tokens`, `cachedContentTokenCount`...). Input and output are estimated at 4 characters per token if missing, cache counts default to 0, and reasoning tokens to the estimate for `reasoning`. |
| `model` | Model name in the response. Default: the requested model. |
| `delayMs` | Wait before responding. |
| `chunkSize` | Characters per streamed chunk. Default: one word per chunk. |
| `chunkDelayMs` | Wait between streamed chunks. |
| `streamError` | `{ afterChunks?, message?, status?, type?, disconnect? }`. Send `afterChunks` content chunks (default 0), then end the stream with the provider's error event, or drop the connection with `disconnect: true`. A request that isn't streamed fails as a whole: an HTTP error with `status` (default 500), or a dropped connection. |

When the request asks for a stream (`stream: true`, or Gemini's `streamGenerateContent`), the same reply is streamed.

## Choosing replies

Each request is answered by the first of these that applies:

1. **Rules**, in the order you added them: `mock.on(match, reply, { times })`. `match` is a substring of the last user message, a RegExp tested against it, or a function that gets the request. Without `times`, a rule answers every matching request.
2. **The queue**: `mock.reply(...replies)`, used in order.
3. **The fallback** from `startMockLLM({ fallback })`: a reply, or `"echo"` to answer `echo: <last user message>`.

If none applies, the server responds with an error that says so, and the SDK throws it:

```
400 mock-llm-server: no reply scripted for request #3 (last user message: "and then?"). Queue one with mock.reply(...), add a rule with mock.on(...), or start the server with a fallback.
```

```js
mock
  .on("weather", "Sunny, 24 °C") // every time the user mentions weather
  .on(/^bye/i, "Goodbye!", { times: 1 })
  .on((req) => req.tools.includes("search"), { toolCalls: [{ name: "search", arguments: { q: "x" } }] })
  .reply("first queued", "second queued");
```

Embeddings, token counting and model lists don't use replies. Embeddings are deterministic unit vectors derived from the input text, so the same text always gets the same vector.

## API

### `startMockLLM(options?)`

Starts the server and resolves to a `MockLLM`. Options:

| Option | Default | What it does |
| --- | --- | --- |
| `port` | `0` | `0` picks a free port. |
| `host` | `"127.0.0.1"` | |
| `replies` | `[]` | Replies to queue right away. |
| `fallback` | none | Reply for unscripted requests, or `"echo"`. |
| `chunkDelayMs` | `0` | Default delay between streamed chunks. |
| `latencyMs` | `0` | Delay before every response. |
| `keepAlive` | `false` | Keep connections open between requests. Off by default so closing the mock never leaves an SDK holding a dead socket. |
| `onRequest` | none | Called with each request as it arrives. |

### `MockLLM`

| Member | What it is |
| --- | --- |
| `url`, `port` | Where it listens. |
| `openaiBaseURL` | `<url>/v1`, for `new OpenAI({ baseURL })`. |
| `anthropicBaseURL` | `<url>`, for `new Anthropic({ baseURL })`. |
| `geminiBaseURL` | `<url>`, for `new GoogleGenAI({ httpOptions: { baseUrl } })`. |
| `reply(...replies)` | Queue replies. Chainable. |
| `on(match, reply, { times? })` | Add a rule. Chainable. |
| `requests`, `lastRequest` | Recorded requests (see below). |
| `pending` | Number of queued replies left. |
| `waitForRequests(count?, { timeoutMs? })` | Resolves with the recorded requests once `count` have arrived in total (default: one more than now). For agents that run in the background. Rejects after `timeoutMs` (default 5000). |
| `assertDone()` | Throws if queued replies were never used. |
| `reset()` | Clear the queue, rules and requests. |
| `close()` | Stop the server. `await using mock = await startMockLLM()` closes it for you. |

Each recorded request has `index`, `provider` (`"openai"`, `"anthropic"` or `"gemini"`), `api` (`"chat.completions"`, `"responses"`, `"messages"`, `"generate_content"`, `"embeddings"` or `"count_tokens"`), `method`, `path`, `headers`, `body` (the parsed JSON), `model`, `stream`, `system`, `messages` (`[{ role, content }]` with tool results as `role: "tool"`), `lastUserMessage` and `tools` (tool names).

### Endpoints

| Endpoint | API |
| --- | --- |
| `POST /v1/chat/completions` | OpenAI Chat Completions, streaming with `stream_options.include_usage` |
| `POST /v1/responses` | OpenAI Responses, streaming events included |
| `POST /v1/embeddings` | OpenAI Embeddings, float or base64, `dimensions` (default 1536) |
| `POST /v1/messages` | Anthropic Messages, streaming events included |
| `POST /v1/messages/count_tokens` | Anthropic token counting |
| `GET /v1/models` | OpenAI's or Anthropic's format, chosen by the `anthropic-version` header |
| `POST /v1beta/models/<model>:generateContent` | Gemini |
| `POST /v1beta/models/<model>:streamGenerateContent?alt=sse` | Gemini streaming |
| `POST /v1beta/models/<model>:countTokens` | Gemini token counting |
| `POST /v1beta/models/<model>:embedContent`, `:batchEmbedContents` | Gemini embeddings, `outputDimensionality` (default 768) |
| `GET /v1beta/models` | Gemini model list |

Anything that speaks these APIs works: LangChain, the Vercel AI SDK, LiteLLM, or your own `fetch` calls.

## Retries

The official SDKs retry 408, 409, 429 and 5xx responses twice by default. Error replies send `retry-after-ms: 0`, so retries happen at once, and each retry uses the next reply. That makes retry logic easy to test:

```js
mock.reply({ error: { status: 503 } }, { error: { status: 429 } }, "Third time lucky");
```

To make an error reach your code on the first try, create the client with `maxRetries: 0`.

## Streams that break

Real streams sometimes fail halfway. `streamError` lets you test what your code does with the part it already received:

```js
mock.reply({ text: "one two three four", streamError: { afterChunks: 2, message: "Upstream died" } });
```

The client gets `"one "` and `"two "`, then the provider's error event: the OpenAI and Anthropic SDKs throw an `APIError` and the Gemini SDK an `ApiError`. With `disconnect: true` the connection closes instead, which SDKs report as a connection error. If `afterChunks` is more than the reply has, the stream still fails before it completes.

## CLI

For tests in other languages, run the server from the command line:

```bash
npx mock-llm-server --port 8787 --script replies.json
```

```
mock-llm-server listening on http://127.0.0.1:8787
  OpenAI base URL:    http://127.0.0.1:8787/v1
  Anthropic base URL: http://127.0.0.1:8787
Press Ctrl+C to stop.
```

`replies.json`:

```json
{
  "replies": ["Hello!", { "toolCalls": [{ "name": "get_weather", "arguments": { "city": "Paris" } }] }],
  "rules": [
    { "match": "weather", "reply": "Sunny, 24 °C" },
    { "match": { "regex": "^bye", "flags": "i" }, "reply": "Goodbye!", "times": 1 }
  ],
  "fallback": "echo"
}
```

While it runs, tests can script it over HTTP:

| Route | What it does |
| --- | --- |
| `POST /__mock/replies` | Queue a reply or an array of replies. |
| `POST /__mock/rules` | Add a rule or an array of rules (same shape as in the script file). |
| `GET /__mock/requests` | List recorded requests. |
| `GET /__mock/wait?count=N&timeoutMs=5000` | Wait until N requests have arrived in total, then list them. `408` on timeout. |
| `POST /__mock/reset` | Clear replies, rules and requests. |

A Python test, for example:

```python
import requests
from openai import OpenAI

requests.post("http://127.0.0.1:8787/__mock/replies", json=["Hi from the mock"])
client = OpenAI(base_url="http://127.0.0.1:8787/v1", api_key="test")
reply = client.chat.completions.create(model="gpt-5", messages=[{"role": "user", "content": "Hi"}])
assert reply.choices[0].message.content == "Hi from the mock"
```

Other flags: `--echo` (echo fallback), `--chunk-delay <ms>`, `--host`, `--quiet`. Run `npx mock-llm-server --help` for everything.

## Not covered

Images, audio, files, batches, the Assistants API, Gemini's Live API and Vertex AI routes aren't implemented. Requests to unknown routes get a 404 that lists what is supported.

## License

MIT
