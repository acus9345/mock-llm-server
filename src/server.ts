import { createServer, type IncomingMessage, type Server, type ServerResponse } from "node:http";
import type { AddressInfo } from "node:net";
import * as anthropic from "./anthropic.js";
import * as gemini from "./gemini.js";
import * as openai from "./openai.js";
import type { Matcher, RecordedRequest, Reply, ReplyObject, ResolvedReply } from "./types.js";
import { estimateTokens, sendJson, sleep, type StreamFailure } from "./util.js";

type Provider = RecordedRequest["provider"];
type ReplyErrorInfo = NonNullable<ResolvedReply["error"]>;

function errorBody(provider: Provider, error: ReplyErrorInfo): Record<string, unknown> {
  if (provider === "anthropic") return anthropic.errorBody(error);
  if (provider === "gemini") return gemini.errorBody(error);
  return openai.errorBody(error);
}

export interface MockOptions {
  /** Port to listen on. Default 0 (a free port). */
  port?: number;
  /** Default `127.0.0.1`. */
  host?: string;
  /** Replies to queue right away. */
  replies?: Reply[];
  /**
   * Used when no rule matches and the queue is empty: a reply, or `"echo"` to answer
   * with `echo: <last user message>`. Default: respond with an error that explains what's missing.
   */
  fallback?: Reply | "echo";
  /** Delay between streamed chunks for every reply, in ms. Default 0. */
  chunkDelayMs?: number;
  /** Delay before every response, in ms. Default 0. */
  latencyMs?: number;
  /**
   * Keep connections open between requests. Default `false`: every response closes its
   * connection, so a closed mock never leaves SDK clients holding dead sockets.
   */
  keepAlive?: boolean;
  /** Called for every model request as soon as it is recorded. */
  onRequest?: (request: RecordedRequest) => void;
}

interface Rule {
  match: Matcher;
  reply: Reply;
  times: number;
}

export interface MockLLM {
  /** `http://127.0.0.1:<port>` */
  readonly url: string;
  readonly port: number;
  /** Pass as `baseURL` to the OpenAI SDK: `<url>/v1`. */
  readonly openaiBaseURL: string;
  /** Pass as `baseURL` to the Anthropic SDK: `<url>`. */
  readonly anthropicBaseURL: string;
  /** Pass as `httpOptions.baseUrl` to the Google Gen AI SDK: `<url>`. */
  readonly geminiBaseURL: string;
  /** Every model request received, oldest first. */
  readonly requests: RecordedRequest[];
  /** The most recent model request. */
  readonly lastRequest: RecordedRequest | undefined;
  /** How many queued replies are left. */
  readonly pending: number;
  /** Queue replies, used in order by requests that no rule matches. */
  reply(...replies: Reply[]): this;
  /** Answer requests that match with `reply`, every time or only `times` times. Rules are checked before the queue. */
  on(match: Matcher, reply: Reply, options?: { times?: number }): this;
  /**
   * Resolve once `count` model requests have arrived in total (default: one more than now), for agents
   * that run in the background. Rejects after `timeoutMs` (default 5000).
   */
  waitForRequests(count?: number, options?: { timeoutMs?: number }): Promise<RecordedRequest[]>;
  /** Throw if queued replies were never used. */
  assertDone(): void;
  /** Clear the queue, rules and recorded requests. */
  reset(): this;
  close(): Promise<void>;
  [Symbol.asyncDispose](): Promise<void>;
}

/** Start a fake OpenAI- and Anthropic-compatible server. */
export async function startMockLLM(options: MockOptions = {}): Promise<MockLLM> {
  const queue: Reply[] = [...(options.replies ?? [])];
  const rules: Rule[] = [];
  const requests: RecordedRequest[] = [];
  const waiters = new Set<() => void>();
  const counters = new Map<string, number>();
  const nextId = (prefix: string) => {
    const n = (counters.get(prefix) ?? 0) + 1;
    counters.set(prefix, n);
    return `${prefix}${n}`;
  };

  async function resolve(req: RecordedRequest): Promise<ResolvedReply | string> {
    let reply: Reply | undefined;
    const ruleIndex = rules.findIndex((r) => matches(r.match, req));
    if (ruleIndex >= 0) {
      const rule = rules[ruleIndex]!;
      reply = rule.reply;
      if (--rule.times <= 0) rules.splice(ruleIndex, 1);
    }
    reply ??= queue.shift();
    if (reply === undefined) {
      if (options.fallback === "echo") reply = `echo: ${req.lastUserMessage}`;
      else if (options.fallback !== undefined) reply = options.fallback;
      else {
        const said = req.lastUserMessage ? ` (last user message: "${req.lastUserMessage.slice(0, 80)}")` : "";
        return (
          `mock-llm-server: no reply scripted for request #${req.index}${said}. ` +
          `Queue one with mock.reply(...), add a rule with mock.on(...), or start the server with a fallback.`
        );
      }
    }
    for (let depth = 0; typeof reply === "function"; depth++) {
      if (depth > 10) throw new Error("mock-llm-server: reply functions nested more than 10 deep");
      reply = await reply(req);
    }
    return normalizeReply(typeof reply === "string" ? { text: reply } : reply, req, nextId, options);
  }

  async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
    const url = new URL(req.url ?? "/", "http://localhost");
    const path = url.pathname.replace(/\/+$/, "");
    const route = path.replace(/^\/v1(?=\/)/, "");
    const geminiRoute = gemini.parseRoute(path);
    const headerProvider: Provider =
      "anthropic-version" in req.headers || "x-api-key" in req.headers
        ? "anthropic"
        : "x-goog-api-key" in req.headers || /^\/v1(beta|alpha)\//.test(path)
          ? "gemini"
          : "openai";

    if (path.startsWith("/__mock/")) return admin(path, url, req, res);
    if (req.method === "GET" && (route === "/models" || /^\/v1(beta|alpha)\/models$/.test(path))) {
      return listModels(res, headerProvider);
    }
    if (req.method !== "POST") return notFound(res, req.method, path, headerProvider);
    if (geminiRoute) return handleGemini(geminiRoute, path, req, res);

    const api =
      route === "/chat/completions"
        ? "chat.completions"
        : route === "/responses"
          ? "responses"
          : route === "/embeddings"
            ? "embeddings"
            : route === "/messages"
              ? "messages"
              : route === "/messages/count_tokens"
                ? "count_tokens"
                : undefined;
    if (!api) return notFound(res, req.method, path, headerProvider);
    const provider = api === "messages" || api === "count_tokens" ? "anthropic" : "openai";

    const body = await readJsonBody(req, res, provider);
    if (!body) return;

    const parsed =
      api === "chat.completions"
        ? openai.parseChat(body)
        : api === "responses"
          ? openai.parseResponses(body)
          : provider === "anthropic"
            ? anthropic.parseMessages(body)
            : { system: "", messages: [], lastUserMessage: "", tools: [] };
    const recorded = record({
      provider,
      api,
      method: req.method ?? "POST",
      path,
      headers: req.headers,
      body,
      model: typeof body.model === "string" ? body.model : "mock-model",
      stream: body.stream === true,
      ...parsed,
    });
    await sleep(options.latencyMs ?? 0);

    if (api === "embeddings") return embeddings(res, body);
    if (api === "count_tokens") {
      return sendJson(res, 200, { input_tokens: estimateTokens(JSON.stringify([parsed.system, parsed.messages, body.tools ?? []])) });
    }

    const reply = await prepareReply(recorded, res);
    if (!reply) return;
    if (api === "chat.completions") return openai.writeChat(res, reply.reply, recorded, nextId("chatcmpl-mock-"), reply.failure);
    if (api === "responses") return openai.writeResponses(res, reply.reply, recorded, nextId("resp_mock_"), nextId, reply.failure);
    return anthropic.writeMessages(res, reply.reply, recorded, nextId("msg_mock_"), reply.failure);
  }

  async function handleGemini(
    route: { model: string; method: string },
    path: string,
    req: IncomingMessage,
    res: ServerResponse,
  ): Promise<void> {
    const generate = route.method === "generateContent" || route.method === "streamGenerateContent";
    const api = generate
      ? "generate_content"
      : route.method === "countTokens"
        ? "count_tokens"
        : route.method === "embedContent" || route.method === "batchEmbedContents"
          ? "embeddings"
          : undefined;
    if (!api) return notFound(res, req.method, path, "gemini");
    const body = await readJsonBody(req, res, "gemini");
    if (!body) return;
    const recorded = record({
      provider: "gemini",
      api,
      method: req.method ?? "POST",
      path,
      headers: req.headers,
      body,
      model: route.model,
      stream: route.method === "streamGenerateContent",
      ...(generate ? gemini.parseGenerate(body) : { system: "", messages: [], lastUserMessage: "", tools: [] }),
    });
    await sleep(options.latencyMs ?? 0);
    if (api === "count_tokens") return sendJson(res, 200, gemini.countTokens(body));
    if (api === "embeddings") return sendJson(res, 200, gemini.embed(route.method, body));
    const reply = await prepareReply(recorded, res);
    if (!reply) return;
    return gemini.writeGenerate(res, reply.reply, recorded, nextId("mock-response-"), reply.failure);
  }

  function record(fields: Omit<RecordedRequest, "index" | "headers"> & { headers: IncomingMessage["headers"] }): RecordedRequest {
    const recorded: RecordedRequest = {
      ...fields,
      index: requests.length + 1,
      headers: Object.fromEntries(Object.entries(fields.headers).map(([k, v]) => [k, Array.isArray(v) ? v.join(", ") : (v ?? "")])),
    };
    requests.push(recorded);
    options.onRequest?.(recorded);
    for (const wake of waiters) wake();
    return recorded;
  }

  /** Pick and resolve the reply, and send it right away if it's an error. Returns what the writers need. */
  async function prepareReply(
    recorded: RecordedRequest,
    res: ServerResponse,
  ): Promise<{ reply: ResolvedReply; failure: StreamFailure | undefined } | undefined> {
    const provider = recorded.provider;
    const reply = await resolve(recorded);
    if (typeof reply === "string") {
      const err = { status: 400, retryAfterMs: 0, message: reply, type: provider === "gemini" ? "FAILED_PRECONDITION" : "mock_no_reply" };
      sendJson(res, 400, errorBody(provider, err), { "x-should-retry": "false" });
      return undefined;
    }
    await sleep(reply.delayMs);
    if (reply.error) {
      const headers = {
        "retry-after-ms": String(reply.error.retryAfterMs),
        "retry-after": String(Math.ceil(reply.error.retryAfterMs / 1000)),
      };
      sendJson(res, reply.error.status, errorBody(provider, reply.error), headers);
      return undefined;
    }
    const streamError = reply.streamError;
    if (!streamError) return { reply, failure: undefined };
    const err = { status: streamError.status, retryAfterMs: 0, message: streamError.message, ...(streamError.type ? { type: streamError.type } : {}) };
    if (!recorded.stream) {
      // A request that isn't streamed fails as a whole.
      if (streamError.disconnect) res.destroy();
      else sendJson(res, err.status, errorBody(provider, err), { "x-should-retry": "false" });
      return undefined;
    }
    const errorEvent =
      provider === "anthropic"
        ? anthropic.streamErrorEvent(err)
        : provider === "gemini"
          ? gemini.streamErrorEvent(err)
          : openai.streamErrorEvent(recorded.api, err);
    return { reply, failure: { afterChunks: streamError.afterChunks, disconnect: streamError.disconnect, errorEvent } };
  }

  async function readJsonBody(req: IncomingMessage, res: ServerResponse, provider: Provider): Promise<Record<string, unknown> | undefined> {
    try {
      const body = JSON.parse((await readBody(req)) || "{}") as unknown;
      if (body === null || typeof body !== "object" || Array.isArray(body)) throw new Error("not an object");
      return body as Record<string, unknown>;
    } catch {
      const err = { status: 400, retryAfterMs: 0, message: "mock-llm-server: request body is not a JSON object." };
      sendJson(res, 400, errorBody(provider, err));
      return undefined;
    }
  }

  function waitForRequests(count: number, timeoutMs: number): Promise<RecordedRequest[]> {
    if (requests.length >= count) return Promise.resolve([...requests]);
    return new Promise((done, fail) => {
      const check = () => {
        if (requests.length < count) return;
        cleanup();
        done([...requests]);
      };
      const timer = setTimeout(() => {
        cleanup();
        fail(new Error(`mock-llm-server: waited ${timeoutMs} ms for ${count} request(s), got ${requests.length}.`));
      }, timeoutMs);
      const cleanup = () => {
        clearTimeout(timer);
        waiters.delete(check);
      };
      waiters.add(check);
    });
  }

  async function admin(path: string, url: URL, req: IncomingMessage, res: ServerResponse): Promise<void> {
    if (req.method === "GET" && path === "/__mock/requests") return sendJson(res, 200, requests);
    if (req.method === "GET" && path === "/__mock/wait") {
      const count = Number(url.searchParams.get("count") ?? requests.length + 1);
      const timeoutMs = Number(url.searchParams.get("timeoutMs") ?? 5000);
      if (!Number.isInteger(count) || count < 1 || !Number.isFinite(timeoutMs) || timeoutMs < 0) {
        return sendJson(res, 400, { error: "count must be a positive integer and timeoutMs a number of milliseconds." });
      }
      try {
        return sendJson(res, 200, await waitForRequests(count, timeoutMs));
      } catch (error) {
        return sendJson(res, 408, { error: (error as Error).message, requests });
      }
    }
    if (req.method === "POST" && path === "/__mock/reset") {
      mock.reset();
      return sendJson(res, 200, { ok: true });
    }
    if (req.method === "POST" && (path === "/__mock/replies" || path === "/__mock/rules")) {
      let body: unknown;
      try {
        body = JSON.parse(await readBody(req));
      } catch {
        return sendJson(res, 400, { error: "Body must be JSON." });
      }
      try {
        if (path === "/__mock/replies") {
          const list = Array.isArray(body) ? body : [body];
          queue.push(...list.map((r) => checkJsonReply(r)));
          return sendJson(res, 200, { pending: queue.length });
        }
        for (const value of Array.isArray(body) ? body : [body]) {
          const rule = parseJsonRule(value);
          mock.on(rule.match, rule.reply, rule.times === undefined ? {} : { times: rule.times });
        }
        return sendJson(res, 200, { rules: rules.length });
      } catch (error) {
        return sendJson(res, 400, { error: (error as Error).message });
      }
    }
    return sendJson(res, 404, {
      error:
        `Unknown admin route ${req.method} ${path}. Use POST /__mock/replies, POST /__mock/rules, GET /__mock/requests, ` +
        `GET /__mock/wait?count=N or POST /__mock/reset.`,
    });
  }

  const server: Server = createServer((req, res) => {
    if (!options.keepAlive) res.shouldKeepAlive = false;
    handle(req, res).catch((error: unknown) => {
      if (!res.headersSent) sendJson(res, 500, { error: { message: `mock-llm-server: ${(error as Error).message}` } });
      else res.destroy();
    });
  });
  await new Promise<void>((done, fail) => {
    server.once("error", fail);
    server.listen(options.port ?? 0, options.host ?? "127.0.0.1", () => {
      server.off("error", fail);
      done();
    });
  });
  const { port, address, family } = server.address() as AddressInfo;
  const host = family === "IPv6" ? `[${address}]` : address;
  const url = `http://${host}:${port}`;

  const mock: MockLLM = {
    url,
    port,
    openaiBaseURL: `${url}/v1`,
    anthropicBaseURL: url,
    geminiBaseURL: url,
    requests,
    get lastRequest() {
      return requests.at(-1);
    },
    get pending() {
      return queue.length;
    },
    reply(...replies) {
      queue.push(...replies);
      return this;
    },
    on(match, reply, opts = {}) {
      rules.push({ match, reply, times: opts.times ?? Infinity });
      return this;
    },
    waitForRequests(count = requests.length + 1, opts = {}) {
      return waitForRequests(count, opts.timeoutMs ?? 5000);
    },
    assertDone() {
      if (queue.length > 0) {
        throw new Error(`mock-llm-server: ${queue.length} queued ${queue.length === 1 ? "reply was" : "replies were"} never used.`);
      }
    },
    reset() {
      queue.length = 0;
      rules.length = 0;
      requests.length = 0;
      counters.clear();
      return this;
    },
    close() {
      return new Promise<void>((done) => {
        server.close(() => done());
        server.closeAllConnections();
      });
    },
    [Symbol.asyncDispose]() {
      return this.close();
    },
  };
  return mock;
}

function matches(match: Matcher, req: RecordedRequest): boolean {
  if (typeof match === "string") return req.lastUserMessage.includes(match);
  if (match instanceof RegExp) {
    match.lastIndex = 0;
    return match.test(req.lastUserMessage);
  }
  return match(req);
}

function normalizeReply(
  reply: ReplyObject,
  req: RecordedRequest,
  nextId: (prefix: string) => string,
  options: MockOptions,
): ResolvedReply {
  const text = reply.json !== undefined ? JSON.stringify(reply.json) : (reply.text ?? "");
  const reasoning = reply.reasoning ?? "";
  const toolCalls = (reply.toolCalls ?? []).map((call) => ({
    id: call.id ?? nextId(req.provider === "anthropic" ? "toolu_mock_" : "call_mock_"),
    name: call.name,
    arguments: typeof call.arguments === "string" ? call.arguments : JSON.stringify(call.arguments ?? {}),
    idGiven: call.id !== undefined,
  }));
  if (req.provider === "anthropic" || req.provider === "gemini") {
    for (const call of toolCalls) {
      try {
        JSON.parse(call.arguments || "{}");
      } catch {
        const name = req.provider === "anthropic" ? "Anthropic" : "Gemini";
        throw new Error(`tool call "${call.name}" has arguments that are not valid JSON, which ${name} can't represent`);
      }
    }
  }
  const error = reply.error
    ? {
        status: reply.error.status ?? 500,
        retryAfterMs: reply.error.retryAfterMs ?? 0,
        ...(reply.error.message === undefined ? {} : { message: reply.error.message }),
        ...(reply.error.type === undefined ? {} : { type: reply.error.type }),
      }
    : undefined;
  const outputText = text + toolCalls.map((c) => c.name + c.arguments).join("");
  const reasoningTokens = reply.usage?.reasoningTokens ?? (reasoning === "" ? 0 : estimateTokens(reasoning));
  const streamError = reply.streamError
    ? {
        afterChunks: reply.streamError.afterChunks ?? 0,
        message: reply.streamError.message ?? "Mock stream error",
        status: reply.streamError.status ?? 500,
        disconnect: reply.streamError.disconnect ?? false,
        ...(reply.streamError.type === undefined ? {} : { type: reply.streamError.type }),
      }
    : undefined;
  return {
    text,
    reasoning,
    toolCalls,
    error,
    stopReason: reply.stopReason ?? (toolCalls.length > 0 ? "tool_calls" : "stop"),
    inputTokens: reply.usage?.inputTokens ?? estimateTokens(JSON.stringify([req.system, req.messages, req.body.tools ?? []])),
    outputTokens: reply.usage?.outputTokens ?? estimateTokens(outputText) + reasoningTokens,
    cachedInputTokens: reply.usage?.cachedInputTokens ?? 0,
    cacheWriteTokens: reply.usage?.cacheWriteTokens ?? 0,
    reasoningTokens,
    model: reply.model ?? req.model,
    delayMs: reply.delayMs ?? 0,
    chunkSize: reply.chunkSize,
    chunkDelayMs: reply.chunkDelayMs ?? options.chunkDelayMs ?? 0,
    streamError,
  };
}

function embeddings(res: ServerResponse, body: Record<string, unknown>): void {
  const input = body.input;
  const items: unknown[] = Array.isArray(input) && input.every((x) => typeof x !== "number") ? input : [input];
  const dims = typeof body.dimensions === "number" ? body.dimensions : 1536;
  const base64 = body.encoding_format === "base64";
  let tokens = 0;
  const data = items.map((item, index) => {
    const text = typeof item === "string" ? item : JSON.stringify(item);
    tokens += estimateTokens(text);
    const vector = openai.fakeEmbedding(text, dims);
    const embedding = base64 ? Buffer.from(new Float32Array(vector).buffer).toString("base64") : vector;
    return { object: "embedding", index, embedding };
  });
  sendJson(res, 200, {
    object: "list",
    data,
    model: typeof body.model === "string" ? body.model : "mock-embedding",
    usage: { prompt_tokens: tokens, total_tokens: tokens },
  });
}

function listModels(res: ServerResponse, provider: Provider): void {
  const ids = ["mock-model", "mock-model-mini"];
  if (provider === "gemini") {
    return sendJson(res, 200, {
      models: ids.map((id) => ({
        name: `models/${id}`,
        displayName: id,
        supportedGenerationMethods: ["generateContent", "countTokens", "embedContent"],
      })),
    });
  }
  if (provider === "anthropic") {
    return sendJson(res, 200, {
      data: ids.map((id) => ({ type: "model", id, display_name: id, created_at: "2026-01-01T00:00:00Z" })),
      has_more: false,
      first_id: ids[0],
      last_id: ids.at(-1),
    });
  }
  sendJson(res, 200, { object: "list", data: ids.map((id) => ({ id, object: "model", created: 1767225600, owned_by: "mock" })) });
}

function notFound(res: ServerResponse, method: string | undefined, path: string, provider: Provider): void {
  const message =
    `mock-llm-server does not handle ${method} ${path}. Supported: POST /v1/chat/completions, POST /v1/responses, ` +
    `POST /v1/embeddings, POST /v1/messages, POST /v1/messages/count_tokens, GET /v1/models, and Gemini's ` +
    `POST /v1beta/models/<model>:generateContent, :streamGenerateContent, :countTokens, :embedContent and :batchEmbedContents.`;
  sendJson(res, 404, errorBody(provider, { status: 404, retryAfterMs: 0, message }));
}

function readBody(req: IncomingMessage): Promise<string> {
  return new Promise((done, fail) => {
    const chunks: Buffer[] = [];
    req.on("data", (c: Buffer) => chunks.push(c));
    req.on("end", () => done(Buffer.concat(chunks).toString("utf8")));
    req.on("error", fail);
  });
}

/** Validate a reply that arrived as JSON (CLI script or admin route). */
export function checkJsonReply(value: unknown): Reply {
  if (typeof value === "string") return value;
  if (value && typeof value === "object" && !Array.isArray(value)) {
    const known = [
      "text",
      "reasoning",
      "json",
      "toolCalls",
      "error",
      "stopReason",
      "usage",
      "model",
      "delayMs",
      "chunkSize",
      "chunkDelayMs",
      "streamError",
    ];
    const unknown = Object.keys(value).filter((k) => !known.includes(k));
    if (unknown.length > 0) throw new Error(`Unknown reply field${unknown.length > 1 ? "s" : ""}: ${unknown.join(", ")}. Known: ${known.join(", ")}.`);
    return value as ReplyObject;
  }
  throw new Error(`A reply must be a string or an object like {"text": "..."}, got ${JSON.stringify(value)}.`);
}

/** Read a rule that arrived as JSON: `{ "match": "text" | { "regex": "...", "flags": "i" }, "reply": ..., "times": 1 }`. */
export function parseJsonRule(value: unknown): { match: Matcher; reply: Reply; times?: number } {
  if (!value || typeof value !== "object") throw new Error(`A rule must be an object like {"match": "weather", "reply": "Sunny"}.`);
  const rule = value as { match?: unknown; reply?: unknown; times?: unknown };
  let match: Matcher;
  if (typeof rule.match === "string") match = rule.match;
  else if (rule.match && typeof rule.match === "object" && typeof (rule.match as { regex?: unknown }).regex === "string") {
    const m = rule.match as { regex: string; flags?: string };
    match = new RegExp(m.regex, m.flags ?? "");
  } else throw new Error(`Rule "match" must be a string or {"regex": "...", "flags": "i"}.`);
  if (rule.times !== undefined && (typeof rule.times !== "number" || rule.times < 1)) {
    throw new Error(`Rule "times" must be a positive number.`);
  }
  const reply = checkJsonReply(rule.reply);
  return typeof rule.times === "number" ? { match, reply, times: rule.times } : { match, reply };
}
