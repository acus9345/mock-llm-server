# mock-llm-server

A fake OpenAI- and Anthropic-compatible API that returns scripted replies, tool calls, errors and streams, so you can unit test agents without API keys. Zero dependencies.

[![npm](https://img.shields.io/npm/v/mock-llm-server)](https://www.npmjs.com/package/mock-llm-server)
[![CI](https://github.com/acus9345/mock-llm-server/actions/workflows/ci.yml/badge.svg)](https://github.com/acus9345/mock-llm-server/actions/workflows/ci.yml)
[![license](https://img.shields.io/npm/l/mock-llm-server)](LICENSE)

- **Real wire formats:** OpenAI Chat Completions, Responses and Embeddings, Anthropic Messages and token counting. Tested against the official `openai` and `@anthropic-ai/sdk` packages, including their streaming helpers.
- **Streaming and tool calls:** text arrives word by word over SSE, and tool-call arguments arrive in pieces, just like the real APIs.
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
mock.reply((req) => `You sent ${req.messages.length} messages to ${req.model}`);
```

| Field | What it does |
| --- | --- |
| `text` | The assistant's text. |
| `json` | Shorthand for `text: JSON.stringify(json)`. |
| `toolCalls` | `[{ name, arguments?, id? }]`. `arguments` is an object or a JSON string. Ids are generated if missing. |
| `error` | `{ status, message?, type?, retryAfterMs? }`. Sent in the provider's error format, so the SDK throws its usual typed error (`RateLimitError` and so on). |
| `stopReason` | `"stop"`, `"length"`, `"tool_calls"` or `"content_filter"`, mapped to each API's wording (`end_turn`, `max_tokens`, `incomplete`...). Inferred if missing. |
| `usage` | `{ inputTokens, outputTokens }`. Estimated at 4 characters per token if missing. |
| `model` | Model name in the response. Default: the requested model. |
| `delayMs` | Wait before responding. |
| `chunkSize` | Characters per streamed chunk. Default: one word per chunk. |
| `chunkDelayMs` | Wait between streamed chunks. |

When the request has `stream: true`, the same reply is streamed.

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
| `reply(...replies)` | Queue replies. Chainable. |
| `on(match, reply, { times? })` | Add a rule. Chainable. |
| `requests`, `lastRequest` | Recorded requests (see below). |
| `pending` | Number of queued replies left. |
| `assertDone()` | Throws if queued replies were never used. |
| `reset()` | Clear the queue, rules and requests. |
| `close()` | Stop the server. `await using mock = await startMockLLM()` closes it for you. |

Each recorded request has `index`, `provider` (`"openai"` or `"anthropic"`), `api` (`"chat.completions"`, `"responses"`, `"messages"`, `"embeddings"` or `"count_tokens"`), `method`, `path`, `headers`, `body` (the parsed JSON), `model`, `stream`, `system`, `messages` (`[{ role, content }]` with tool results as `role: "tool"`), `lastUserMessage` and `tools` (tool names).

### Endpoints

| Endpoint | API |
| --- | --- |
| `POST /v1/chat/completions` | OpenAI Chat Completions, streaming with `stream_options.include_usage` |
| `POST /v1/responses` | OpenAI Responses, streaming events included |
| `POST /v1/embeddings` | OpenAI Embeddings, float or base64, `dimensions` (default 1536) |
| `POST /v1/messages` | Anthropic Messages, streaming events included |
| `POST /v1/messages/count_tokens` | Anthropic token counting |
| `GET /v1/models` | Either provider's format, chosen by the `anthropic-version` header |

Anything that speaks these APIs works: LangChain, the Vercel AI SDK, LiteLLM, or your own `fetch` calls.

## Retries

The official SDKs retry 408, 409, 429 and 5xx responses twice by default. Error replies send `retry-after-ms: 0`, so retries happen at once, and each retry uses the next reply. That makes retry logic easy to test:

```js
mock.reply({ error: { status: 503 } }, { error: { status: 429 } }, "Third time lucky");
```

To make an error reach your code on the first try, create the client with `maxRetries: 0`.

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

Images, audio, files, batches, the Assistants API and Gemini's native API aren't implemented. Requests to unknown routes get a 404 that lists what is supported.

## License

MIT
