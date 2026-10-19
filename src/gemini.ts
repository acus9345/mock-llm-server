import type { ServerResponse } from "node:http";
import type { RecordedRequest, ResolvedReply } from "./types.js";
import { chunkText, contentText, estimateTokens, sendJson, SseWriter, type StreamFailure } from "./util.js";
import { fakeEmbedding } from "./openai.js";

type Parsed = Pick<RecordedRequest, "system" | "messages" | "lastUserMessage" | "tools">;

/** `/v1beta/models/gemini-x:generateContent` → `{ model: "gemini-x", method: "generateContent" }` */
export function parseRoute(path: string): { model: string; method: string } | undefined {
  const m = /^\/(?:v1|v1beta|v1alpha)\/models\/([^/:]+):([A-Za-z]+)$/.exec(path);
  return m ? { model: decodeURIComponent(m[1]!), method: m[2]! } : undefined;
}

export function parseGenerate(body: Record<string, unknown>): Parsed {
  const system = partsText(body.systemInstruction ?? body.system_instruction);
  const messages: { role: string; content: string }[] = [];
  const contents = Array.isArray(body.contents) ? (body.contents as Record<string, unknown>[]) : [];
  for (const content of contents) {
    const role = content.role === "model" ? "assistant" : String(content.role ?? "user");
    const parts = Array.isArray(content.parts) ? (content.parts as Record<string, unknown>[]) : [];
    const text = parts
      .filter((p) => typeof p.text === "string" && p.thought !== true)
      .map((p) => p.text as string)
      .join("\n");
    for (const p of parts) {
      const call = p.functionCall as Record<string, unknown> | undefined;
      const result = p.functionResponse as Record<string, unknown> | undefined;
      if (call) messages.push({ role: "assistant", content: `${String(call.name)}(${JSON.stringify(call.args ?? {})})` });
      if (result) messages.push({ role: "tool", content: contentText(result.response) });
    }
    if (text !== "" || !parts.some((p) => p.functionCall || p.functionResponse)) messages.push({ role, content: text });
  }
  const last = contents.at(-1);
  const lastUserMessage =
    last && (last.role ?? "user") === "user" && messages.at(-1)?.role === "user" ? messages.at(-1)!.content : "";
  const tools: string[] = [];
  for (const tool of Array.isArray(body.tools) ? (body.tools as Record<string, unknown>[]) : []) {
    const declarations = tool.functionDeclarations ?? tool.function_declarations;
    if (Array.isArray(declarations)) tools.push(...declarations.map((d: Record<string, unknown>) => String(d.name)));
    else tools.push(...Object.keys(tool));
  }
  return { system, messages, lastUserMessage, tools };
}

function partsText(value: unknown): string {
  if (typeof value === "string") return value;
  if (value && typeof value === "object" && Array.isArray((value as { parts?: unknown }).parts)) {
    return ((value as { parts: Record<string, unknown>[] }).parts)
      .map((p) => (typeof p.text === "string" ? p.text : ""))
      .filter(Boolean)
      .join("\n");
  }
  return contentText(value);
}

const FINISH: Record<ResolvedReply["stopReason"], string> = {
  stop: "STOP",
  length: "MAX_TOKENS",
  tool_calls: "STOP",
  content_filter: "SAFETY",
};

export async function writeGenerate(
  res: ServerResponse,
  reply: ResolvedReply,
  req: RecordedRequest,
  id: string,
  failure?: StreamFailure,
): Promise<void> {
  const callParts = reply.toolCalls.map((c) => ({
    functionCall: { ...(c.idGiven ? { id: c.id } : {}), name: c.name, args: JSON.parse(c.arguments || "{}") as unknown },
  }));
  const usageMetadata = {
    promptTokenCount: reply.inputTokens,
    candidatesTokenCount: Math.max(0, reply.outputTokens - reply.reasoningTokens),
    totalTokenCount: reply.inputTokens + reply.outputTokens,
    ...(reply.reasoningTokens > 0 ? { thoughtsTokenCount: reply.reasoningTokens } : {}),
    ...(reply.cachedInputTokens > 0 ? { cachedContentTokenCount: reply.cachedInputTokens } : {}),
  };
  const response = (parts: unknown[], final: boolean) => ({
    candidates: [
      {
        content: { role: "model", parts },
        ...(final ? { finishReason: FINISH[reply.stopReason] } : {}),
        index: 0,
      },
    ],
    ...(final ? { usageMetadata } : {}),
    modelVersion: reply.model,
    responseId: id,
  });

  if (!req.stream) {
    const parts: unknown[] = [];
    if (reply.reasoning !== "") parts.push({ text: reply.reasoning, thought: true });
    if (reply.text !== "" || callParts.length === 0) parts.push({ text: reply.text });
    parts.push(...callParts);
    sendJson(res, 200, response(parts, true));
    return;
  }

  // streamGenerateContent?alt=sse: every event is a whole GenerateContentResponse holding the new parts.
  const sse = new SseWriter(res, reply.chunkDelayMs, {}, failure);
  for (const text of chunkText(reply.reasoning, reply.chunkSize)) {
    await sse.send(response([{ text, thought: true }], false), undefined, "delta");
  }
  for (const text of chunkText(reply.text, reply.chunkSize)) await sse.send(response([{ text }], false), undefined, "delta");
  // Gemini sends function calls whole, not in pieces.
  for (const part of callParts) await sse.send(response([part], false), undefined, "delta");
  await sse.send(response(reply.text === "" && callParts.length === 0 ? [{ text: "" }] : [], true), undefined, "final");
  sse.end();
}

const STATUS: Record<number, string> = {
  400: "INVALID_ARGUMENT",
  401: "UNAUTHENTICATED",
  403: "PERMISSION_DENIED",
  404: "NOT_FOUND",
  409: "ABORTED",
  429: "RESOURCE_EXHAUSTED",
  499: "CANCELLED",
  500: "INTERNAL",
  503: "UNAVAILABLE",
  504: "DEADLINE_EXCEEDED",
};

export function errorBody(error: NonNullable<ResolvedReply["error"]>): Record<string, unknown> {
  return {
    error: {
      code: error.status,
      message: error.message ?? `Mock error ${error.status}`,
      status: error.type ?? STATUS[error.status] ?? (error.status >= 500 ? "INTERNAL" : "INVALID_ARGUMENT"),
    },
  };
}

/** Gemini ends a failed stream with a bare JSON error object, not an SSE event. */
export function streamErrorEvent(error: NonNullable<ResolvedReply["error"]>): string {
  return JSON.stringify(errorBody(error));
}

export function countTokens(body: Record<string, unknown>): Record<string, unknown> {
  const request = (body.generateContentRequest ?? body) as Record<string, unknown>;
  const parsed = parseGenerate(request);
  return { totalTokens: estimateTokens(JSON.stringify([parsed.system, parsed.messages, request.tools ?? []])) };
}

/** `embedContent` → `{ embedding }`, `batchEmbedContents` → `{ embeddings }`. Deterministic, like the OpenAI endpoint. */
export function embed(method: string, body: Record<string, unknown>): Record<string, unknown> {
  const one = (request: Record<string, unknown>) => {
    const dims = typeof request.outputDimensionality === "number" ? request.outputDimensionality : 768;
    return { values: fakeEmbedding(partsText(request.content), dims) };
  };
  if (method === "batchEmbedContents") {
    const requests = Array.isArray(body.requests) ? (body.requests as Record<string, unknown>[]) : [];
    return { embeddings: requests.map(one) };
  }
  return { embedding: one(body) };
}

