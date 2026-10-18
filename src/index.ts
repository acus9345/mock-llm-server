/**
 * A fake OpenAI- and Anthropic-compatible API server for testing agents without API keys.
 *
 * @example
 * import OpenAI from "openai";
 * import { startMockLLM } from "mock-llm-server";
 *
 * const mock = await startMockLLM();
 * mock.reply("Hello from the mock!");
 * const openai = new OpenAI({ baseURL: mock.openaiBaseURL, apiKey: "test" });
 * const res = await openai.chat.completions.create({ model: "gpt-5", messages: [{ role: "user", content: "Hi" }] });
 * await mock.close();
 */
export { checkJsonReply, parseJsonRule, startMockLLM } from "./server.js";
export type { MockLLM, MockOptions } from "./server.js";
export type { Matcher, RecordedRequest, Reply, ReplyError, ReplyObject, ToolCall } from "./types.js";
