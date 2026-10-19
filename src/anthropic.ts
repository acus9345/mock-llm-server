import type { ServerResponse } from "node:http";
import type { RecordedRequest, ResolvedReply } from "./types.js";
import { chunkText, contentText, sendJson, sseEvent, SseWriter, type StreamFailure } from "./util.js";

type Parsed = Pick<RecordedRequest, "system" | "messages" | "lastUserMessage" | "tools">;

export function parseMessages(body: Record<string, unknown>): Parsed {
  const system = contentText(body.system);
  const messages: { role: string; content: string }[] = [];
  let lastUserMessage = "";
  const raw = Array.isArray(body.messages) ? (body.messages as Record<string, unknown>[]) : [];
  for (const [i, m] of raw.entries()) {
    const parts = Array.isArray(m.content) ? (m.content as Record<string, unknown>[]) : [{ type: "text", text: m.content }];
    const text = parts
      .filter((p) => p.type === "text")
      .map((p) => contentText(p.text))
      .join("\n");
    for (const p of parts) {
      if (p.type === "tool_result") messages.push({ role: "tool", content: contentText(p.content) });
      if (p.type === "tool_use") messages.push({ role: "assistant", content: `${String(p.name)}(${JSON.stringify(p.input)})` });
    }
    if (text !== "" || !parts.some((p) => p.type === "tool_result" || p.type === "tool_use")) {
      messages.push({ role: String(m.role), content: text });
    }
    if (i === raw.length - 1 && m.role === "user") lastUserMessage = text;
  }
  const tools = (Array.isArray(body.tools) ? body.tools : []).map((t: Record<string, unknown>) => String(t.name ?? t.type));
  return { system, messages, lastUserMessage, tools };
}

const STOP: Record<ResolvedReply["stopReason"], string> = {
  stop: "end_turn",
  length: "max_tokens",
  tool_calls: "tool_use",
  content_filter: "refusal",
};

export async function writeMessages(
  res: ServerResponse,
  reply: ResolvedReply,
  req: RecordedRequest,
  id: string,
  failure?: StreamFailure,
): Promise<void> {
  const blocks: Record<string, unknown>[] = [];
  const signature = Buffer.from(`mock-signature:${id}`).toString("base64");
  if (reply.reasoning !== "") blocks.push({ type: "thinking", thinking: reply.reasoning, signature });
  if (reply.text !== "" || reply.toolCalls.length === 0) blocks.push({ type: "text", text: reply.text });
  for (const call of reply.toolCalls) {
    blocks.push({ type: "tool_use", id: call.id, name: call.name, input: JSON.parse(call.arguments || "{}") });
  }
  const usage = {
    input_tokens: reply.inputTokens,
    output_tokens: reply.outputTokens,
    cache_creation_input_tokens: reply.cacheWriteTokens,
    cache_read_input_tokens: reply.cachedInputTokens,
    service_tier: "standard",
  };
  const message = (content: unknown[], stop: string | null, u: object) => ({
    id,
    type: "message",
    role: "assistant",
    model: reply.model,
    content,
    stop_reason: stop,
    stop_sequence: null,
    usage: u,
  });

  if (!req.stream) {
    sendJson(res, 200, message(blocks, STOP[reply.stopReason], usage), { "request-id": `req_${id}` });
    return;
  }

  const sse = new SseWriter(res, reply.chunkDelayMs, { "request-id": `req_${id}` }, failure);
  await sse.send({ type: "message_start", message: message([], null, { ...usage, output_tokens: 1 }) }, "message_start");
  for (const [index, block] of blocks.entries()) {
    if (index === 0) await sse.send({ type: "ping" }, "ping");
    if (block.type === "thinking") {
      await sse.send(
        { type: "content_block_start", index, content_block: { type: "thinking", thinking: "", signature: "" } },
        "content_block_start",
      );
      for (const thinking of chunkText(reply.reasoning, reply.chunkSize)) {
        await sse.send({ type: "content_block_delta", index, delta: { type: "thinking_delta", thinking } }, "content_block_delta", "delta");
      }
      await sse.send({ type: "content_block_delta", index, delta: { type: "signature_delta", signature } }, "content_block_delta");
    } else if (block.type === "text") {
      await sse.send({ type: "content_block_start", index, content_block: { type: "text", text: "" } }, "content_block_start");
      for (const text of chunkText(reply.text, reply.chunkSize)) {
        await sse.send({ type: "content_block_delta", index, delta: { type: "text_delta", text } }, "content_block_delta", "delta");
      }
    } else {
      await sse.send(
        { type: "content_block_start", index, content_block: { type: "tool_use", id: block.id, name: block.name, input: {} } },
        "content_block_start",
      );
      const json = JSON.stringify(block.input);
      for (const partial_json of chunkText(json, reply.chunkSize ?? 8)) {
        await sse.send(
          { type: "content_block_delta", index, delta: { type: "input_json_delta", partial_json } },
          "content_block_delta",
          "delta",
        );
      }
    }
    await sse.send({ type: "content_block_stop", index }, "content_block_stop");
  }
  await sse.send(
    {
      type: "message_delta",
      delta: { stop_reason: STOP[reply.stopReason], stop_sequence: null },
      usage: { output_tokens: reply.outputTokens },
    },
    "message_delta",
    "final",
  );
  await sse.send({ type: "message_stop" }, "message_stop");
  sse.end();
}

const ERROR_TYPES: Record<number, string> = {
  400: "invalid_request_error",
  401: "authentication_error",
  402: "billing_error",
  403: "permission_error",
  404: "not_found_error",
  413: "request_too_large",
  429: "rate_limit_error",
  500: "api_error",
  504: "timeout_error",
  529: "overloaded_error",
};

/** The SSE event that ends a Messages stream with an error. */
export function streamErrorEvent(error: NonNullable<ResolvedReply["error"]>): string {
  return sseEvent(errorBody(error), "error");
}

export function errorBody(error: NonNullable<ResolvedReply["error"]>): Record<string, unknown> {
  return {
    type: "error",
    error: {
      type: error.type ?? ERROR_TYPES[error.status] ?? (error.status >= 500 ? "api_error" : "invalid_request_error"),
      message: error.message ?? `Mock error ${error.status}`,
    },
  };
}
