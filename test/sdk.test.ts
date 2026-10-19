import Anthropic from "@anthropic-ai/sdk";
import assert from "node:assert/strict";
import { after, before, beforeEach, describe, it } from "node:test";
import OpenAI from "openai";
import { startMockLLM, type MockLLM } from "../dist/index.js";
import { loopbackDeliversLateWrites } from "./helpers.ts";

let mock: MockLLM;
let openai: OpenAI;
let anthropic: Anthropic;

before(async () => {
  mock = await startMockLLM();
  openai = new OpenAI({ baseURL: mock.openaiBaseURL, apiKey: "test", maxRetries: 0 });
  anthropic = new Anthropic({ baseURL: mock.anthropicBaseURL, apiKey: "test", maxRetries: 0 });
});
after(() => mock.close());
beforeEach(() => mock.reset());

const user = (content: string) => [{ role: "user" as const, content }];
const weatherTool = {
  type: "function" as const,
  function: { name: "get_weather", parameters: { type: "object", properties: { city: { type: "string" } } } },
};

describe("OpenAI Chat Completions", () => {
  it("returns a scripted reply and records the request", async () => {
    mock.reply("Hello there!");
    const res = await openai.chat.completions.create({
      model: "gpt-5",
      messages: [{ role: "system", content: "Be nice" }, ...user("Hi")],
    });
    assert.equal(res.choices[0]?.message.content, "Hello there!");
    assert.equal(res.choices[0]?.finish_reason, "stop");
    assert.equal(res.model, "gpt-5");
    assert.ok(res.usage!.prompt_tokens > 0);
    assert.equal(mock.lastRequest?.api, "chat.completions");
    assert.equal(mock.lastRequest?.system, "Be nice");
    assert.equal(mock.lastRequest?.lastUserMessage, "Hi");
    assert.equal(mock.lastRequest?.headers.authorization, "Bearer test");
  });

  it("returns tool calls", async () => {
    mock.reply({ toolCalls: [{ name: "get_weather", arguments: { city: "Paris" } }] });
    const res = await openai.chat.completions.create({ model: "gpt-5", messages: user("Weather?"), tools: [weatherTool] });
    const call = res.choices[0]?.message.tool_calls?.[0];
    assert.equal(res.choices[0]?.finish_reason, "tool_calls");
    assert.equal(res.choices[0]?.message.content, null);
    assert.ok(call?.type === "function");
    assert.equal(call.function.name, "get_weather");
    assert.deepEqual(JSON.parse(call.function.arguments), { city: "Paris" });
    assert.deepEqual(mock.lastRequest?.tools, ["get_weather"]);
  });

  it("streams text and tool calls that the SDK can reassemble", async () => {
    mock.reply({ text: "Let me check the weather.", toolCalls: [{ name: "get_weather", arguments: { city: "Oslo" } }] });
    const stream = openai.chat.completions.stream({
      model: "gpt-5",
      messages: user("Weather in Oslo?"),
      tools: [weatherTool],
      stream_options: { include_usage: true },
    });
    const deltas: string[] = [];
    stream.on("content.delta", ({ delta }) => deltas.push(delta));
    const final = await stream.finalChatCompletion();
    assert.deepEqual(deltas, ["Let ", "me ", "check ", "the ", "weather."]);
    assert.equal(final.choices[0]?.message.content, "Let me check the weather.");
    const call = final.choices[0]?.message.tool_calls?.[0];
    assert.ok(call?.type === "function");
    assert.deepEqual(JSON.parse(call.function.arguments), { city: "Oslo" });
    assert.equal(final.choices[0]?.finish_reason, "tool_calls");
    assert.ok(final.usage?.total_tokens);
  });

  it("returns JSON for structured output", async () => {
    mock.reply({ json: { name: "Ada", age: 36 } });
    const res = await openai.chat.completions.create({ model: "gpt-5", messages: user("Who?") });
    assert.deepEqual(JSON.parse(res.choices[0]!.message.content!), { name: "Ada", age: 36 });
  });

  it("returns errors the SDK turns into typed exceptions", async () => {
    mock.reply({ error: { status: 429, message: "Slow down" } });
    await assert.rejects(
      openai.chat.completions.create({ model: "gpt-5", messages: user("Hi") }),
      (e: unknown) => e instanceof OpenAI.RateLimitError && e.message.includes("Slow down"),
    );
  });

  it("lets the SDK retry without waiting", async () => {
    const retrying = new OpenAI({ baseURL: mock.openaiBaseURL, apiKey: "test", maxRetries: 2 });
    mock.reply({ error: { status: 503 } }, { error: { status: 429 } }, "Finally");
    const started = Date.now();
    const res = await retrying.chat.completions.create({ model: "gpt-5", messages: user("Hi") });
    assert.equal(res.choices[0]?.message.content, "Finally");
    assert.equal(mock.requests.length, 3);
    assert.ok(Date.now() - started < 2000);
  });

  it("fails clearly when nothing is scripted", async () => {
    await assert.rejects(
      openai.chat.completions.create({ model: "gpt-5", messages: user("Hello?") }),
      /no reply scripted for request #1 \(last user message: "Hello\?"\)/,
    );
  });
});

describe("OpenAI Responses", () => {
  it("returns text and function calls", async () => {
    mock.reply({ text: "Checking.", toolCalls: [{ name: "get_weather", arguments: { city: "Rome" } }] });
    const res = await openai.responses.create({
      model: "gpt-5",
      instructions: "Be brief",
      input: "Weather in Rome?",
      tools: [{ type: "function", name: "get_weather", parameters: { type: "object", properties: {} }, strict: false }],
    });
    assert.equal(res.output_text, "Checking.");
    const call = res.output.find((o) => o.type === "function_call");
    assert.ok(call?.type === "function_call");
    assert.equal(call.name, "get_weather");
    assert.deepEqual(JSON.parse(call.arguments), { city: "Rome" });
    assert.equal(mock.lastRequest?.system, "Be brief");
    assert.equal(mock.lastRequest?.lastUserMessage, "Weather in Rome?");
  });

  it("streams events the SDK helper accepts", async () => {
    mock.reply({ text: "Streaming works", toolCalls: [{ name: "lookup", arguments: { q: "x" } }] });
    const stream = openai.responses.stream({
      model: "gpt-5",
      input: [{ role: "user", content: [{ type: "input_text", text: "Go" }] }],
    });
    const deltas: string[] = [];
    stream.on("response.output_text.delta", (e) => deltas.push(e.delta));
    const final = await stream.finalResponse();
    assert.deepEqual(deltas, ["Streaming ", "works"]);
    assert.equal(final.output_text, "Streaming works");
    assert.equal(final.status, "completed");
    const call = final.output.find((o) => o.type === "function_call");
    assert.ok(call?.type === "function_call");
    assert.equal(call.arguments, '{"q":"x"}');
  });

  it("reports stopReason length as incomplete", async () => {
    mock.reply({ text: "Cut", stopReason: "length" });
    const res = await openai.responses.create({ model: "gpt-5", input: "Long story" });
    assert.equal(res.status, "incomplete");
    assert.equal(res.incomplete_details?.reason, "max_output_tokens");
  });
});

describe("OpenAI embeddings and models", () => {
  it("returns deterministic unit vectors, as floats or base64", async () => {
    const a = await openai.embeddings.create({ model: "text-embedding-3-small", input: ["cat", "dog"], dimensions: 8 });
    const b = await openai.embeddings.create({ model: "text-embedding-3-small", input: "cat", dimensions: 8, encoding_format: "float" });
    assert.equal(a.data.length, 2);
    assert.equal(a.data[0]!.embedding.length, 8);
    for (const [i, x] of a.data[0]!.embedding.entries()) assert.ok(Math.abs(x - b.data[0]!.embedding[i]!) < 1e-6);
    assert.ok(Math.abs(Math.hypot(...b.data[0]!.embedding) - 1) < 1e-4);
    assert.notDeepEqual(a.data[0]!.embedding, a.data[1]!.embedding);
    assert.equal(mock.pending, 0);
  });

  it("lists models", async () => {
    const page = await openai.models.list();
    assert.deepEqual(
      page.data.map((m) => m.id),
      ["mock-model", "mock-model-mini"],
    );
  });
});

describe("Anthropic Messages", () => {
  it("returns text", async () => {
    mock.reply("Bonjour");
    const msg = await anthropic.messages.create({
      model: "claude-sonnet-5-5",
      max_tokens: 100,
      system: "Speak French",
      messages: user("Hello"),
    });
    assert.deepEqual(msg.content, [{ type: "text", text: "Bonjour" }]);
    assert.equal(msg.stop_reason, "end_turn");
    assert.equal(mock.lastRequest?.provider, "anthropic");
    assert.equal(mock.lastRequest?.system, "Speak French");
    assert.equal(mock.lastRequest?.headers["x-api-key"], "test");
  });

  it("returns tool use and records tool results in the next turn", async () => {
    mock.reply({ toolCalls: [{ name: "get_weather", arguments: { city: "Lima" }, id: "toolu_1" }] }, "It is 20 C in Lima.");
    const tools = [{ name: "get_weather", input_schema: { type: "object" as const, properties: {} } }];
    const first = await anthropic.messages.create({ model: "c", max_tokens: 100, tools, messages: user("Weather?") });
    assert.equal(first.stop_reason, "tool_use");
    assert.deepEqual(first.content, [{ type: "tool_use", id: "toolu_1", name: "get_weather", input: { city: "Lima" } }]);
    const second = await anthropic.messages.create({
      model: "c",
      max_tokens: 100,
      tools,
      messages: [
        ...user("Weather?"),
        { role: "assistant", content: first.content },
        { role: "user", content: [{ type: "tool_result", tool_use_id: "toolu_1", content: "20 C" }] },
      ],
    });
    assert.equal(second.content[0]?.type === "text" && second.content[0].text, "It is 20 C in Lima.");
    assert.equal(mock.lastRequest?.lastUserMessage, "");
    assert.deepEqual(mock.lastRequest?.messages.at(-1), { role: "tool", content: "20 C" });
  });

  it("streams events the SDK's MessageStream accepts", async () => {
    mock.reply({ text: "Here you go", toolCalls: [{ name: "save", arguments: { path: "a.txt", body: "hi" } }] });
    const stream = anthropic.messages.stream({ model: "c", max_tokens: 100, messages: user("Save it") });
    const texts: string[] = [];
    stream.on("text", (t) => texts.push(t));
    const final = await stream.finalMessage();
    assert.deepEqual(texts, ["Here ", "you ", "go"]);
    assert.equal(final.stop_reason, "tool_use");
    const tool = final.content.find((c) => c.type === "tool_use");
    assert.ok(tool?.type === "tool_use");
    assert.deepEqual(tool.input, { path: "a.txt", body: "hi" });
    assert.ok(final.usage.output_tokens > 1);
  });

  it("returns typed errors and counts tokens", async () => {
    mock.reply({ error: { status: 529, message: "Overloaded" } });
    await assert.rejects(
      anthropic.messages.create({ model: "c", max_tokens: 10, messages: user("Hi") }),
      (e: unknown) => e instanceof Anthropic.APIError && e.status === 529 && /Overloaded/.test(e.message),
    );
    const count = await anthropic.messages.countTokens({ model: "c", messages: user("Count these words please") });
    assert.ok(count.input_tokens > 0);
    const models = await anthropic.models.list();
    assert.equal(models.data[0]?.id, "mock-model");
  });
});

describe("scripting", () => {
  it("checks rules before the queue and honours times", async () => {
    mock.on("weather", "Sunny").on(/^bye/i, "Goodbye!", { times: 1 }).reply("queued 1", "queued 2");
    const ask = async (content: string) =>
      (await openai.chat.completions.create({ model: "m", messages: user(content) })).choices[0]?.message.content;
    assert.equal(await ask("weather today?"), "Sunny");
    assert.equal(await ask("Bye now"), "Goodbye!");
    assert.equal(await ask("Bye again"), "queued 1");
    assert.equal(await ask("weather tomorrow?"), "Sunny");
    assert.equal(mock.pending, 1);
    assert.throws(() => mock.assertDone(), /1 queued reply was never used/);
  });

  it("accepts reply functions that see the request", async () => {
    mock.on((r) => r.tools.length > 0, (r) => ({ toolCalls: [{ name: r.tools[0]!, arguments: {} }] }));
    mock.reply((r) => `You said ${r.messages.length} message(s) to ${r.model}`);
    const plain = await openai.chat.completions.create({ model: "m1", messages: user("x") });
    assert.equal(plain.choices[0]?.message.content, "You said 1 message(s) to m1");
    const tooly = await openai.chat.completions.create({ model: "m1", messages: user("x"), tools: [weatherTool] });
    assert.equal(tooly.choices[0]?.message.tool_calls?.[0]?.type, "function");
  });

  it("echo fallback", async () => {
    const echo = await startMockLLM({ fallback: "echo" });
    try {
      const client = new OpenAI({ baseURL: echo.openaiBaseURL, apiKey: "x", maxRetries: 0 });
      const res = await client.chat.completions.create({ model: "m", messages: user("ping") });
      assert.equal(res.choices[0]?.message.content, "echo: ping");
    } finally {
      await echo.close();
    }
  });

  it("chunkSize and chunkDelayMs", async (t) => {
    if (!(await loopbackDeliversLateWrites())) {
      t.skip("this machine's loopback drops TCP writes sent after a delay (a VPN or firewall filter?)");
      return;
    }
    const echo = await startMockLLM();
    try {
      const client = new OpenAI({ baseURL: echo.openaiBaseURL, apiKey: "x", maxRetries: 0 });
      echo.reply({ text: "abcdef", chunkSize: 2, chunkDelayMs: 15 });
      const stream = await client.chat.completions.create({ model: "m", messages: user("x"), stream: true });
      const parts: string[] = [];
      const started = Date.now();
      for await (const chunk of stream) if (chunk.choices[0]?.delta.content) parts.push(chunk.choices[0].delta.content);
      assert.deepEqual(parts, ["ab", "cd", "ef"]);
      assert.ok(Date.now() - started >= 40);
    } finally {
      await echo.close();
    }
  });

  it("can be scripted over HTTP", async () => {
    const post = (path: string, body: unknown) =>
      fetch(`${mock.url}${path}`, { method: "POST", body: JSON.stringify(body) }).then((r) => r.json());
    assert.deepEqual(await post("/__mock/replies", ["one", { text: "two" }]), { pending: 2 });
    assert.deepEqual(await post("/__mock/rules", { match: { regex: "^hey", flags: "i" }, reply: "hi!" }), { rules: 1 });
    const bad = await fetch(`${mock.url}/__mock/replies`, { method: "POST", body: JSON.stringify({ txt: "typo" }) });
    assert.equal(bad.status, 400);
    assert.match(((await bad.json()) as { error: string }).error, /Unknown reply field: txt/);
    const res = await openai.chat.completions.create({ model: "m", messages: user("Hey") });
    assert.equal(res.choices[0]?.message.content, "hi!");
    const requests = (await fetch(`${mock.url}/__mock/requests`).then((r) => r.json())) as unknown[];
    assert.equal(requests.length, 1);
    await post("/__mock/reset", {});
    assert.equal(mock.pending, 0);
  });

  it("returns a helpful 404 for unknown routes", async () => {
    const res = await fetch(`${mock.url}/v1/images/generations`, { method: "POST", body: "{}" });
    assert.equal(res.status, 404);
    assert.match(((await res.json()) as { error: { message: string } }).error.message, /does not handle POST \/v1\/images\/generations/);
  });
});
