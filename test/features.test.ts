import Anthropic from "@anthropic-ai/sdk";
import { ApiError, GoogleGenAI } from "@google/genai";
import assert from "node:assert/strict";
import { after, before, beforeEach, describe, it } from "node:test";
import OpenAI from "openai";
import { startMockLLM, type MockLLM } from "../dist/index.js";
import { loopbackDeliversLateWrites } from "./helpers.ts";

let mock: MockLLM;
let openai: OpenAI;
let anthropic: Anthropic;
let gemini: GoogleGenAI;

before(async () => {
  mock = await startMockLLM();
  openai = new OpenAI({ baseURL: mock.openaiBaseURL, apiKey: "test", maxRetries: 0 });
  anthropic = new Anthropic({ baseURL: mock.anthropicBaseURL, apiKey: "test", maxRetries: 0 });
  gemini = new GoogleGenAI({ apiKey: "test", httpOptions: { baseUrl: mock.geminiBaseURL } });
});
after(() => mock.close());
beforeEach(() => mock.reset());

const user = (content: string) => [{ role: "user" as const, content }];
const weatherDeclaration = {
  name: "get_weather",
  parametersJsonSchema: { type: "object", properties: { city: { type: "string" } } },
};

describe("Gemini", () => {
  it("returns text and records system instructions, tools and messages", async () => {
    mock.reply("Hallo!");
    const res = await gemini.models.generateContent({
      model: "gemini-2.5-flash",
      contents: "Say hi in German",
      config: { systemInstruction: "Be brief", tools: [{ functionDeclarations: [weatherDeclaration] }] },
    });
    assert.equal(res.text, "Hallo!");
    assert.equal(res.candidates?.[0]?.finishReason, "STOP");
    assert.ok((res.usageMetadata?.promptTokenCount ?? 0) > 0);
    const req = mock.lastRequest!;
    assert.equal(req.provider, "gemini");
    assert.equal(req.api, "generate_content");
    assert.equal(req.model, "gemini-2.5-flash");
    assert.equal(req.system, "Be brief");
    assert.equal(req.lastUserMessage, "Say hi in German");
    assert.deepEqual(req.tools, ["get_weather"]);
    assert.equal(req.headers["x-goog-api-key"], "test");
  });

  it("returns function calls and reads function responses in the next turn", async () => {
    mock.reply({ toolCalls: [{ name: "get_weather", arguments: { city: "Oslo" } }] }, "4 °C in Oslo.");
    const first = await gemini.models.generateContent({ model: "g", contents: "Weather in Oslo?" });
    assert.deepEqual(first.functionCalls, [{ name: "get_weather", args: { city: "Oslo" } }]);
    const second = await gemini.models.generateContent({
      model: "g",
      contents: [
        { role: "user", parts: [{ text: "Weather in Oslo?" }] },
        { role: "model", parts: [{ functionCall: { name: "get_weather", args: { city: "Oslo" } } }] },
        { role: "user", parts: [{ functionResponse: { name: "get_weather", response: { temp: 4 } } }] },
      ],
    });
    assert.equal(second.text, "4 °C in Oslo.");
    assert.deepEqual(mock.lastRequest?.messages.slice(-2), [
      { role: "assistant", content: 'get_weather({"city":"Oslo"})' },
      { role: "tool", content: '{"temp":4}' },
    ]);
    assert.equal(mock.lastRequest?.lastUserMessage, "");
  });

  it("streams text chunk by chunk", async () => {
    mock.reply({ text: "One two three", toolCalls: [{ name: "save", arguments: { a: 1 }, id: "call-7" }] });
    const stream = await gemini.models.generateContentStream({ model: "g", contents: "Count" });
    const texts: string[] = [];
    const calls: unknown[] = [];
    let finish: string | undefined;
    for await (const chunk of stream) {
      if (chunk.text) texts.push(chunk.text);
      calls.push(...(chunk.functionCalls ?? []));
      finish ??= chunk.candidates?.[0]?.finishReason;
    }
    assert.deepEqual(texts, ["One ", "two ", "three"]);
    assert.deepEqual(calls, [{ id: "call-7", name: "save", args: { a: 1 } }]);
    assert.equal(finish, "STOP");
    assert.equal(mock.lastRequest?.stream, true);
  });

  it("returns errors the SDK throws as ApiError", async () => {
    mock.reply({ error: { status: 429, message: "Quota exceeded" } });
    await assert.rejects(
      gemini.models.generateContent({ model: "g", contents: "Hi" }),
      (e: unknown) => e instanceof ApiError && e.status === 429 && /RESOURCE_EXHAUSTED/.test(e.message),
    );
  });

  it("counts tokens, embeds deterministically and lists models", async () => {
    const count = await gemini.models.countTokens({ model: "g", contents: "Count these words please" });
    assert.ok((count.totalTokens ?? 0) > 0);
    const a = await gemini.models.embedContent({ model: "e", contents: ["cat", "dog"], config: { outputDimensionality: 8 } });
    const b = await gemini.models.embedContent({ model: "e", contents: "cat", config: { outputDimensionality: 8 } });
    assert.equal(a.embeddings?.length, 2);
    assert.equal(a.embeddings?.[0]?.values?.length, 8);
    assert.deepEqual(a.embeddings?.[0]?.values, b.embeddings?.[0]?.values);
    assert.notDeepEqual(a.embeddings?.[0]?.values, a.embeddings?.[1]?.values);
    const pager = await gemini.models.list();
    assert.equal(pager.page[0]?.name, "models/mock-model");
    assert.equal(mock.pending, 0);
  });
});

describe("reasoning", () => {
  it("Anthropic: thinking blocks, plain and streamed", async () => {
    mock.reply({ reasoning: "The user wants a greeting.", text: "Hello!" });
    const msg = await anthropic.messages.create({ model: "c", max_tokens: 100, messages: user("Hi") });
    assert.equal(msg.content[0]?.type, "thinking");
    assert.ok(msg.content[0]?.type === "thinking" && msg.content[0].thinking === "The user wants a greeting.");
    assert.ok(msg.content[0]?.type === "thinking" && msg.content[0].signature.length > 0);
    assert.deepEqual(msg.content[1], { type: "text", text: "Hello!" });

    mock.reply({ reasoning: "Think hard", text: "Done" });
    const stream = anthropic.messages.stream({ model: "c", max_tokens: 100, messages: user("Go") });
    const thoughts: string[] = [];
    stream.on("thinking", (delta) => thoughts.push(delta));
    const final = await stream.finalMessage();
    assert.deepEqual(thoughts, ["Think ", "hard"]);
    const thinking = final.content[0];
    assert.ok(thinking?.type === "thinking" && thinking.thinking === "Think hard" && thinking.signature.length > 0);
    assert.equal(final.content[1]?.type === "text" && final.content[1].text, "Done");
  });

  it("OpenAI Responses: a reasoning item with a summary, plain and streamed", async () => {
    mock.reply({ reasoning: "Check the city first.", text: "Rome." });
    const res = await openai.responses.create({ model: "o", input: "Capital of Italy?" });
    assert.deepEqual(res.output[0], { id: "rs_mock_1", type: "reasoning", summary: [{ type: "summary_text", text: "Check the city first." }] });
    assert.equal(res.output_text, "Rome.");
    assert.ok((res.usage?.output_tokens_details.reasoning_tokens ?? 0) > 0);

    mock.reply({ reasoning: "Step one", text: "Answer" });
    const stream = openai.responses.stream({ model: "o", input: "Go" });
    const deltas: string[] = [];
    stream.on("response.reasoning_summary_text.delta", (e) => deltas.push(e.delta));
    const final = await stream.finalResponse();
    assert.deepEqual(deltas, ["Step ", "one"]);
    const item = final.output[0];
    assert.ok(item?.type === "reasoning");
    assert.deepEqual(item.summary, [{ type: "summary_text", text: "Step one" }]);
    assert.equal(final.output_text, "Answer");
  });

  it("Chat Completions: reasoning_content, as DeepSeek-style servers send it", async () => {
    mock.reply({ reasoning: "Short thought", text: "Hi" });
    const res = await openai.chat.completions.create({ model: "deepseek-reasoner", messages: user("Hi") });
    assert.equal((res.choices[0]?.message as unknown as { reasoning_content: string }).reasoning_content, "Short thought");
    assert.equal(res.choices[0]?.message.content, "Hi");

    mock.reply({ reasoning: "Short thought", text: "Hi" });
    const stream = await openai.chat.completions.create({ model: "deepseek-reasoner", messages: user("Hi"), stream: true });
    let reasoning = "";
    let content = "";
    for await (const chunk of stream) {
      const delta = chunk.choices[0]?.delta as { reasoning_content?: string; content?: string | null } | undefined;
      reasoning += delta?.reasoning_content ?? "";
      content += delta?.content ?? "";
    }
    assert.equal(reasoning, "Short thought");
    assert.equal(content, "Hi");
  });

  it("Gemini: thought parts that .text leaves out", async () => {
    mock.reply({ reasoning: "Thinking...", text: "Answer", usage: { reasoningTokens: 7 } });
    const res = await gemini.models.generateContent({ model: "g", contents: "Q" });
    assert.equal(res.text, "Answer");
    assert.deepEqual(res.candidates?.[0]?.content?.parts?.[0], { text: "Thinking...", thought: true });
    assert.equal(res.usageMetadata?.thoughtsTokenCount, 7);
  });
});

describe("usage", () => {
  it("sends cache and reasoning token counts in each provider's fields", async () => {
    const usage = { inputTokens: 100, outputTokens: 40, cachedInputTokens: 80, cacheWriteTokens: 15, reasoningTokens: 30 };
    mock.reply({ text: "a", usage }, { text: "b", usage }, { text: "c", usage }, { text: "d", usage });
    const chat = await openai.chat.completions.create({ model: "m", messages: user("x") });
    assert.equal(chat.usage?.prompt_tokens, 100);
    assert.equal(chat.usage?.prompt_tokens_details?.cached_tokens, 80);
    assert.equal(chat.usage?.completion_tokens_details?.reasoning_tokens, 30);
    const resp = await openai.responses.create({ model: "m", input: "x" });
    assert.equal(resp.usage?.input_tokens_details.cached_tokens, 80);
    assert.equal(resp.usage?.output_tokens_details.reasoning_tokens, 30);
    const msg = await anthropic.messages.create({ model: "m", max_tokens: 10, messages: user("x") });
    assert.equal(msg.usage.input_tokens, 100);
    assert.equal(msg.usage.cache_read_input_tokens, 80);
    assert.equal(msg.usage.cache_creation_input_tokens, 15);
    const gem = await gemini.models.generateContent({ model: "g", contents: "x" });
    assert.equal(gem.usageMetadata?.cachedContentTokenCount, 80);
    assert.equal(gem.usageMetadata?.candidatesTokenCount, 10);
    assert.equal(gem.usageMetadata?.totalTokenCount, 140);
  });
});

describe("streamError", () => {
  it("OpenAI Chat Completions: the SDK throws after the chunks that arrived", async () => {
    mock.reply({ text: "one two three four", streamError: { afterChunks: 2, message: "Upstream died" } });
    const stream = await openai.chat.completions.create({ model: "m", messages: user("x"), stream: true });
    const parts: string[] = [];
    await assert.rejects(
      (async () => {
        for await (const chunk of stream) if (chunk.choices[0]?.delta.content) parts.push(chunk.choices[0].delta.content);
      })(),
      (e: unknown) => e instanceof OpenAI.APIError && /Upstream died/.test(e.message),
    );
    assert.deepEqual(parts, ["one ", "two "]);
  });

  it("OpenAI Responses: an error event", async () => {
    mock.reply({ text: "one two", streamError: { afterChunks: 1, message: "Responses broke" } });
    const stream = openai.responses.stream({ model: "m", input: "x" });
    await assert.rejects(stream.finalResponse(), /Responses broke/);
  });

  it("Anthropic: an error event the SDK throws as an APIError", async () => {
    mock.reply({ text: "one two", streamError: { afterChunks: 1, status: 529, message: "Overloaded mid-stream" } });
    const stream = anthropic.messages.stream({ model: "c", max_tokens: 10, messages: user("x") });
    await assert.rejects(stream.finalMessage(), (e: unknown) => e instanceof Anthropic.APIError && /Overloaded mid-stream/.test(e.message));
  });

  it("Gemini: the SDK's stream throws", async () => {
    mock.reply({ text: "one two three", streamError: { afterChunks: 1, message: "Gemini broke" } });
    const stream = await gemini.models.generateContentStream({ model: "g", contents: "x" });
    await assert.rejects(async () => {
      for await (const _ of stream);
    });
  });

  it("a failure set after more chunks than the reply has still happens before the end", async () => {
    mock.reply({ text: "short", streamError: { afterChunks: 99, message: "Late failure" } });
    const stream = await openai.chat.completions.create({ model: "m", messages: user("x"), stream: true });
    await assert.rejects(async () => {
      for await (const _ of stream);
    }, /Late failure/);
  });

  it("disconnect drops the connection mid-stream", async (t) => {
    if (!(await loopbackDeliversLateWrites())) {
      t.skip("this machine's loopback drops TCP segments after the first, including the close (a VPN or firewall filter?)");
      return;
    }
    mock.reply({ text: "one two three", streamError: { afterChunks: 1, disconnect: true } });
    const parts: string[] = [];
    await assert.rejects(async () => {
      const stream = await openai.chat.completions.create({ model: "m", messages: user("x"), stream: true });
      for await (const chunk of stream) if (chunk.choices[0]?.delta.content) parts.push(chunk.choices[0].delta.content);
    });
    assert.deepEqual(parts, ["one "]);
  });

  it("disconnect drops a request that isn't streamed", async () => {
    mock.reply({ text: "one", streamError: { disconnect: true } });
    await assert.rejects(openai.chat.completions.create({ model: "m", messages: user("x") }), (e: unknown) => e instanceof OpenAI.APIConnectionError);
  });

  it("a request that isn't streamed fails with the status", async () => {
    mock.reply({ text: "one", streamError: { status: 503, message: "Gone" } });
    await assert.rejects(
      anthropic.messages.create({ model: "c", max_tokens: 10, messages: user("x") }),
      (e: unknown) => e instanceof Anthropic.APIError && e.status === 503,
    );
  });
});

describe("waitForRequests", () => {
  it("resolves when requests arrive and times out otherwise", async () => {
    mock.reply("Background answer");
    const pending = openai.chat.completions.create({ model: "m", messages: user("from the background") });
    const [first] = await mock.waitForRequests(1);
    assert.equal(first?.lastUserMessage, "from the background");
    await pending;
    await assert.rejects(mock.waitForRequests(2, { timeoutMs: 50 }), /waited 50 ms for 2 request\(s\), got 1/);
  });

  it("works over HTTP for other languages", async () => {
    mock.reply("x");
    const waiting = fetch(`${mock.url}/__mock/wait?count=1`).then((r) => r.json() as Promise<{ lastUserMessage: string }[]>);
    await openai.chat.completions.create({ model: "m", messages: user("hello") });
    assert.equal((await waiting)[0]?.lastUserMessage, "hello");
    const late = await fetch(`${mock.url}/__mock/wait?count=5&timeoutMs=20`);
    assert.equal(late.status, 408);
  });
});
