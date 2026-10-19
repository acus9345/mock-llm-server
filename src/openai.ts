import type { ServerResponse } from "node:http";
import type { RecordedRequest, ResolvedReply } from "./types.js";
import { chunkText, contentText, sendJson, sseEvent, SseWriter, unixSeconds, type StreamFailure } from "./util.js";

type Parsed = Pick<RecordedRequest, "system" | "messages" | "lastUserMessage" | "tools">;

export function parseChat(body: Record<string, unknown>): Parsed {
  const raw = Array.isArray(body.messages) ? (body.messages as Record<string, unknown>[]) : [];
  const system = raw
    .filter((m) => m.role === "system" || m.role === "developer")
    .map((m) => contentText(m.content))
    .join("\n");
  const messages = raw
    .filter((m) => m.role !== "system" && m.role !== "developer")
    .map((m) => ({ role: String(m.role), content: contentText(m.content) }));
  const lastUser = [...messages].reverse().find((m) => m.role === "user");
  const tools = (Array.isArray(body.tools) ? body.tools : []).map((t: Record<string, unknown>) =>
    String((t.function as Record<string, unknown> | undefined)?.name ?? t.name ?? t.type),
  );
  return { system, messages, lastUserMessage: messages.at(-1)?.role === "user" ? (lastUser?.content ?? "") : "", tools };
}

export function parseResponses(body: Record<string, unknown>): Parsed {
  const system = typeof body.instructions === "string" ? body.instructions : "";
  const messages: { role: string; content: string }[] = [];
  if (typeof body.input === "string") messages.push({ role: "user", content: body.input });
  else if (Array.isArray(body.input)) {
    for (const item of body.input as Record<string, unknown>[]) {
      if (item.type === "function_call_output") messages.push({ role: "tool", content: contentText(item.output) });
      else if (item.type === "function_call") messages.push({ role: "assistant", content: `${item.name}(${item.arguments})` });
      else if (typeof item.role === "string") messages.push({ role: item.role, content: contentText(item.content) });
    }
  }
  const sys = messages.filter((m) => m.role === "system" || m.role === "developer").map((m) => m.content);
  const convo = messages.filter((m) => m.role !== "system" && m.role !== "developer");
  const tools = (Array.isArray(body.tools) ? body.tools : []).map((t: Record<string, unknown>) => String(t.name ?? t.type));
  return {
    system: [system, ...sys].filter(Boolean).join("\n"),
    messages: convo,
    lastUserMessage: convo.at(-1)?.role === "user" ? convo.at(-1)!.content : "",
    tools,
  };
}

const FINISH: Record<ResolvedReply["stopReason"], string> = {
  stop: "stop",
  length: "length",
  tool_calls: "tool_calls",
  content_filter: "content_filter",
};

export async function writeChat(
  res: ServerResponse,
  reply: ResolvedReply,
  req: RecordedRequest,
  id: string,
  failure?: StreamFailure,
): Promise<void> {
  const created = unixSeconds();
  const usage = {
    prompt_tokens: reply.inputTokens,
    completion_tokens: reply.outputTokens,
    total_tokens: reply.inputTokens + reply.outputTokens,
    prompt_tokens_details: { cached_tokens: reply.cachedInputTokens, audio_tokens: 0 },
    completion_tokens_details: {
      reasoning_tokens: reply.reasoningTokens,
      audio_tokens: 0,
      accepted_prediction_tokens: 0,
      rejected_prediction_tokens: 0,
    },
  };
  const toolCalls = reply.toolCalls.map((t) => ({ id: t.id, type: "function", function: { name: t.name, arguments: t.arguments } }));

  if (!req.stream) {
    sendJson(res, 200, {
      id,
      object: "chat.completion",
      created,
      model: reply.model,
      choices: [
        {
          index: 0,
          message: {
            role: "assistant",
            content: reply.text === "" && toolCalls.length > 0 ? null : reply.text,
            // Not part of OpenAI's API, but what DeepSeek, vLLM and similar servers send.
            ...(reply.reasoning === "" ? {} : { reasoning_content: reply.reasoning }),
            refusal: null,
            annotations: [],
            ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}),
          },
          logprobs: null,
          finish_reason: FINISH[reply.stopReason],
        },
      ],
      usage,
      service_tier: "default",
      system_fingerprint: "fp_mock",
    });
    return;
  }

  const sse = new SseWriter(res, reply.chunkDelayMs, {}, failure);
  const base = { id, object: "chat.completion.chunk", created, model: reply.model, service_tier: "default", system_fingerprint: "fp_mock" };
  const chunk = (delta: Record<string, unknown>, finish: string | null = null) => ({
    ...base,
    choices: [{ index: 0, delta, logprobs: null, finish_reason: finish }],
  });
  await sse.send(chunk({ role: "assistant", content: "", refusal: null }));
  for (const piece of chunkText(reply.reasoning, reply.chunkSize)) {
    await sse.send(chunk({ reasoning_content: piece }), undefined, "delta");
  }
  for (const piece of chunkText(reply.text, reply.chunkSize)) await sse.send(chunk({ content: piece }), undefined, "delta");
  for (const [index, call] of reply.toolCalls.entries()) {
    await sse.send(chunk({ tool_calls: [{ index, id: call.id, type: "function", function: { name: call.name, arguments: "" } }] }));
    for (const piece of chunkText(call.arguments, reply.chunkSize ?? 8)) {
      await sse.send(chunk({ tool_calls: [{ index, function: { arguments: piece } }] }), undefined, "delta");
    }
  }
  await sse.send(chunk({}, FINISH[reply.stopReason]), undefined, "final");
  const options = req.body.stream_options as { include_usage?: boolean } | undefined;
  if (options?.include_usage) await sse.send({ ...base, choices: [], usage });
  await sse.send("[DONE]");
  sse.end();
}

export async function writeResponses(
  res: ServerResponse,
  reply: ResolvedReply,
  req: RecordedRequest,
  id: string,
  nextId: (prefix: string) => string,
  failure?: StreamFailure,
): Promise<void> {
  const created = unixSeconds();
  const incomplete = reply.stopReason === "length" || reply.stopReason === "content_filter";
  const items: Record<string, unknown>[] = [];
  if (reply.reasoning !== "") {
    items.push({ id: nextId("rs_mock_"), type: "reasoning", summary: [{ type: "summary_text", text: reply.reasoning }] });
  }
  const messageId = nextId("msg_mock_");
  if (reply.text !== "" || reply.toolCalls.length === 0) {
    items.push({
      id: messageId,
      type: "message",
      status: incomplete ? "incomplete" : "completed",
      role: "assistant",
      content: [{ type: "output_text", text: reply.text, annotations: [], logprobs: [] }],
    });
  }
  for (const call of reply.toolCalls) {
    items.push({
      id: nextId("fc_mock_"),
      type: "function_call",
      status: "completed",
      call_id: call.id,
      name: call.name,
      arguments: call.arguments,
    });
  }
  const body = req.body;
  const response = (status: string, output: unknown[]) => ({
    id,
    object: "response",
    created_at: created,
    status,
    background: false,
    error: null,
    incomplete_details:
      status === "incomplete" ? { reason: reply.stopReason === "length" ? "max_output_tokens" : "content_filter" } : null,
    instructions: body.instructions ?? null,
    max_output_tokens: body.max_output_tokens ?? null,
    model: reply.model,
    output,
    parallel_tool_calls: body.parallel_tool_calls ?? true,
    previous_response_id: body.previous_response_id ?? null,
    reasoning: { effort: null, summary: null },
    store: body.store ?? true,
    temperature: body.temperature ?? 1,
    text: body.text ?? { format: { type: "text" } },
    tool_choice: body.tool_choice ?? "auto",
    tools: body.tools ?? [],
    top_p: body.top_p ?? 1,
    truncation: "disabled",
    usage:
      status === "in_progress"
        ? null
        : {
            input_tokens: reply.inputTokens,
            input_tokens_details: { cached_tokens: reply.cachedInputTokens },
            output_tokens: reply.outputTokens,
            output_tokens_details: { reasoning_tokens: reply.reasoningTokens },
            total_tokens: reply.inputTokens + reply.outputTokens,
          },
    user: body.user ?? null,
    metadata: body.metadata ?? {},
  });
  const finalStatus = incomplete ? "incomplete" : "completed";

  if (!req.stream) {
    sendJson(res, 200, response(finalStatus, items));
    return;
  }

  const sse = new SseWriter(res, reply.chunkDelayMs, {}, failure);
  let seq = 0;
  const emit = (type: string, data: Record<string, unknown>, kind?: "delta" | "final") =>
    sse.send({ type, sequence_number: seq++, ...data }, type, kind);
  await emit("response.created", { response: response("in_progress", []) });
  await emit("response.in_progress", { response: response("in_progress", []) });
  for (const [outputIndex, item] of items.entries()) {
    if (item.type === "reasoning") {
      await emit("response.output_item.added", { output_index: outputIndex, item: { ...item, summary: [] } });
      const part = { type: "summary_text", text: "" };
      const at = { item_id: item.id, output_index: outputIndex, summary_index: 0 };
      await emit("response.reasoning_summary_part.added", { ...at, part });
      for (const delta of chunkText(reply.reasoning, reply.chunkSize)) {
        await emit("response.reasoning_summary_text.delta", { ...at, delta }, "delta");
      }
      await emit("response.reasoning_summary_text.done", { ...at, text: reply.reasoning });
      await emit("response.reasoning_summary_part.done", { ...at, part: { ...part, text: reply.reasoning } });
    } else if (item.type === "message") {
      await emit("response.output_item.added", {
        output_index: outputIndex,
        item: { ...item, status: "in_progress", content: [] },
      });
      const part = { type: "output_text", text: "", annotations: [], logprobs: [] };
      await emit("response.content_part.added", { item_id: item.id, output_index: outputIndex, content_index: 0, part });
      for (const delta of chunkText(reply.text, reply.chunkSize)) {
        await emit(
          "response.output_text.delta",
          { item_id: item.id, output_index: outputIndex, content_index: 0, delta, logprobs: [] },
          "delta",
        );
      }
      await emit("response.output_text.done", {
        item_id: item.id,
        output_index: outputIndex,
        content_index: 0,
        text: reply.text,
        logprobs: [],
      });
      await emit("response.content_part.done", {
        item_id: item.id,
        output_index: outputIndex,
        content_index: 0,
        part: { ...part, text: reply.text },
      });
    } else {
      const args = String(item.arguments);
      await emit("response.output_item.added", {
        output_index: outputIndex,
        item: { ...item, status: "in_progress", arguments: "" },
      });
      for (const delta of chunkText(args, reply.chunkSize ?? 8)) {
        await emit("response.function_call_arguments.delta", { item_id: item.id, output_index: outputIndex, delta }, "delta");
      }
      await emit("response.function_call_arguments.done", {
        item_id: item.id,
        output_index: outputIndex,
        name: item.name,
        arguments: args,
      });
    }
    await emit("response.output_item.done", { output_index: outputIndex, item });
  }
  await emit(incomplete ? "response.incomplete" : "response.completed", { response: response(finalStatus, items) }, "final");
  sse.end();
}

export function errorBody(error: NonNullable<ResolvedReply["error"]>): Record<string, unknown> {
  const status = error.status;
  const type =
    error.type ??
    (status === 429
      ? "rate_limit_exceeded"
      : status === 401
        ? "invalid_request_error"
        : status >= 500
          ? "server_error"
          : "invalid_request_error");
  const code = status === 429 ? "rate_limit_exceeded" : status === 401 ? "invalid_api_key" : null;
  return {
    error: {
      message: error.message ?? `Mock error ${status}`,
      type,
      param: null,
      code,
    },
  };
}

/** The SSE event that ends a Chat Completions or Responses stream with an error. */
export function streamErrorEvent(api: string, error: NonNullable<ResolvedReply["error"]>): string {
  const body = (errorBody(error) as { error: Record<string, unknown> }).error;
  if (api === "chat.completions") return sseEvent({ error: body });
  return sseEvent({ type: "error", code: body.code ?? body.type, message: body.message, param: null, sequence_number: 0 }, "error");
}

/** Deterministic, unit-length fake embedding for a string. */
export function fakeEmbedding(text: string, dimensions: number): number[] {
  // FNV-1a seed, then xorshift.
  let h = 0x811c9dc5;
  for (let i = 0; i < text.length; i++) {
    h ^= text.charCodeAt(i);
    h = Math.imul(h, 0x01000193) >>> 0;
  }
  let x = h || 1;
  const v: number[] = [];
  for (let i = 0; i < dimensions; i++) {
    x ^= x << 13;
    x >>>= 0;
    x ^= x >>> 17;
    x ^= x << 5;
    x >>>= 0;
    v.push((x / 0xffffffff) * 2 - 1);
  }
  const norm = Math.hypot(...v) || 1;
  return v.map((n) => Number((n / norm).toFixed(6)));
}
