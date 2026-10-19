import type { ServerResponse } from "node:http";

/** Split text into stream chunks: one word (with its trailing space) per chunk, or fixed-size pieces. */
export function chunkText(text: string, size: number | undefined): string[] {
  if (text === "") return [];
  if (size !== undefined && size > 0) {
    const out: string[] = [];
    const chars = Array.from(text);
    for (let i = 0; i < chars.length; i += size) out.push(chars.slice(i, i + size).join(""));
    return out;
  }
  return text.match(/\s*\S+\s*|\s+/g) ?? [text];
}

export function estimateTokens(text: string): number {
  return Math.max(1, Math.ceil(text.length / 4));
}

export const sleep = (ms: number) => (ms > 0 ? new Promise<void>((r) => setTimeout(r, ms)) : Promise.resolve());

export function sendJson(res: ServerResponse, status: number, body: unknown, headers: Record<string, string> = {}): void {
  const text = JSON.stringify(body);
  res.writeHead(status, {
    "content-type": "application/json",
    "content-length": Buffer.byteLength(text),
    ...headers,
  });
  res.end(text);
}

/** How a stream fails partway: after `afterChunks` deltas, write `errorEvent` (or drop the connection). */
export interface StreamFailure {
  afterChunks: number;
  disconnect: boolean;
  /** The raw text to write as the provider's error event. */
  errorEvent: string;
}

/** Server-sent events writer with optional delay between events and a scripted failure. */
export class SseWriter {
  #res: ServerResponse;
  #delay: number;
  #first = true;
  #failure: StreamFailure | undefined;
  #deltas = 0;
  #stopped = false;
  constructor(res: ServerResponse, delayMs: number, headers: Record<string, string> = {}, failure?: StreamFailure) {
    this.#res = res;
    this.#delay = delayMs;
    this.#failure = failure;
    res.writeHead(200, {
      "content-type": "text/event-stream",
      "cache-control": "no-cache",
      ...headers,
    });
  }
  /**
   * Send one event. Content deltas pass `"delta"` so a scripted failure can count them; the first
   * event that finishes the response passes `"final"`, so a failure with more chunks than the reply
   * has still happens before the response completes.
   */
  async send(data: unknown, event?: string, kind?: "delta" | "final"): Promise<void> {
    if (this.#stopped) return;
    const delta = kind === "delta";
    if (this.#failure && (kind === "final" || (delta && this.#deltas >= this.#failure.afterChunks))) return this.#fail(this.#failure);
    if (!this.#first) await sleep(this.#delay);
    this.#first = false;
    if (this.#res.destroyed) return;
    if (delta) this.#deltas++;
    const payload = typeof data === "string" ? data : JSON.stringify(data);
    this.#res.write(`${event ? `event: ${event}\n` : ""}data: ${payload}\n\n`);
  }
  end(): void {
    if (!this.#stopped) this.#res.end();
    this.#stopped = true;
  }
  #fail(failure: StreamFailure): void {
    this.#stopped = true;
    // Ending the socket (not destroying it) flushes the chunks already written, then closes the
    // connection mid-response, the way a crashed upstream looks to the client.
    if (failure.disconnect) this.#res.write("", () => this.#res.destroy());
    else this.#res.end(failure.errorEvent);
  }
}

/** Flatten OpenAI/Anthropic content (string or array of parts) to text. */
export function contentText(content: unknown): string {
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return content == null ? "" : JSON.stringify(content);
  return content
    .map((part: unknown) => {
      if (typeof part === "string") return part;
      if (part && typeof part === "object") {
        const p = part as Record<string, unknown>;
        if (typeof p.text === "string") return p.text;
        if (p.type === "tool_result") return contentText(p.content);
        if (typeof p.output === "string") return p.output;
      }
      return "";
    })
    .filter(Boolean)
    .join("\n");
}

/** Format an SSE event. */
export function sseEvent(data: unknown, event?: string): string {
  return `${event ? `event: ${event}\n` : ""}data: ${JSON.stringify(data)}\n\n`;
}

export function unixSeconds(): number {
  return Math.floor(Date.now() / 1000);
}
