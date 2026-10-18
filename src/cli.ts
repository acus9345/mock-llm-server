#!/usr/bin/env node
import { readFileSync } from "node:fs";
import { parseArgs } from "node:util";
import { checkJsonReply, parseJsonRule, startMockLLM, type MockOptions, type Reply } from "./index.js";

const HELP = `mock-llm-server - fake OpenAI and Anthropic API for tests

Usage:
  mock-llm-server [--port 8787] [--script replies.json] [options]

Starts a server that answers like the OpenAI (Chat Completions, Responses,
Embeddings) and Anthropic (Messages) APIs with replies you script, including
streaming and tool calls. Point your SDK's base URL at it:
  OpenAI:    <url>/v1   (OPENAI_BASE_URL)
  Anthropic: <url>      (ANTHROPIC_BASE_URL)

Options:
  -p, --port <n>          Port to listen on. Default 8787 (0 picks a free port).
      --host <host>       Default 127.0.0.1.
  -s, --script <file>     JSON file with "replies", "rules" and/or "fallback".
      --echo              Answer unscripted requests with "echo: <last user message>".
      --chunk-delay <ms>  Delay between streamed chunks.
  -q, --quiet             Don't log requests.
  -h, --help              Show this help.
  -v, --version           Show the version number.

Script file example:
  {
    "replies": ["Hello!", { "toolCalls": [{ "name": "get_weather", "arguments": { "city": "Paris" } }] }],
    "rules": [{ "match": "weather", "reply": "Sunny, 24 C" },
              { "match": { "regex": "^bye", "flags": "i" }, "reply": "Goodbye!", "times": 1 }],
    "fallback": "echo"
  }

While it runs, tests in any language can script it over HTTP:
  POST /__mock/replies   queue a reply or an array of replies
  POST /__mock/rules     add a rule or an array of rules
  GET  /__mock/requests  list recorded requests
  POST /__mock/reset     clear replies, rules and requests
`;

class UsageError extends Error {}

function version(): string {
  const pkg = JSON.parse(readFileSync(new URL("../package.json", import.meta.url), "utf8")) as { version: string };
  return pkg.version;
}

function loadScript(path: string): Pick<MockOptions, "replies" | "fallback"> & { rules: ReturnType<typeof parseJsonRule>[] } {
  let text: string;
  try {
    text = readFileSync(path, "utf8");
  } catch (error) {
    const code = (error as NodeJS.ErrnoException).code;
    throw new UsageError(code === "ENOENT" ? `can't find script "${path}".` : `can't read "${path}": ${(error as Error).message}`);
  }
  let json: unknown;
  try {
    json = JSON.parse(text);
  } catch (error) {
    throw new UsageError(`"${path}" is not valid JSON: ${(error as Error).message}`);
  }
  if (!json || typeof json !== "object" || Array.isArray(json)) {
    throw new UsageError(`"${path}" must contain an object with "replies", "rules" and/or "fallback".`);
  }
  const { replies, rules, fallback, ...rest } = json as Record<string, unknown>;
  if (Object.keys(rest).length > 0) throw new UsageError(`"${path}" has unknown keys: ${Object.keys(rest).join(", ")}.`);
  try {
    const out: Pick<MockOptions, "replies" | "fallback"> & { rules: ReturnType<typeof parseJsonRule>[] } = {
      replies: (Array.isArray(replies) ? replies : replies === undefined ? [] : [replies]).map(checkJsonReply),
      rules: (Array.isArray(rules) ? rules : rules === undefined ? [] : [rules]).map(parseJsonRule),
    };
    if (fallback !== undefined) out.fallback = fallback === "echo" ? "echo" : (checkJsonReply(fallback) as Reply);
    return out;
  } catch (error) {
    throw new UsageError(`${path}: ${(error as Error).message}`);
  }
}

async function main(argv: string[]): Promise<number> {
  let values;
  try {
    ({ values } = parseArgs({
      args: argv,
      options: {
        port: { type: "string", short: "p" },
        host: { type: "string" },
        script: { type: "string", short: "s" },
        echo: { type: "boolean" },
        "chunk-delay": { type: "string" },
        quiet: { type: "boolean", short: "q" },
        help: { type: "boolean", short: "h" },
        version: { type: "boolean", short: "v" },
      },
    }));
  } catch (error) {
    throw new UsageError((error as Error).message);
  }
  if (values.help) {
    process.stdout.write(HELP);
    return 0;
  }
  if (values.version) {
    process.stdout.write(`${version()}\n`);
    return 0;
  }
  const port = Number(values.port ?? process.env.PORT ?? 8787);
  if (!Number.isInteger(port) || port < 0 || port > 65535) throw new UsageError(`--port must be 0-65535, not "${values.port}".`);
  const chunkDelay = values["chunk-delay"] === undefined ? 0 : Number(values["chunk-delay"]);
  if (!Number.isFinite(chunkDelay) || chunkDelay < 0) throw new UsageError(`--chunk-delay must be a number of milliseconds.`);

  const script = values.script ? loadScript(values.script) : { rules: [] };
  const options: MockOptions = { port, chunkDelayMs: chunkDelay };
  if (values.host) options.host = values.host;
  if (script.replies) options.replies = script.replies;
  if (values.echo) options.fallback = "echo";
  else if (script.fallback !== undefined) options.fallback = script.fallback;

  if (!values.quiet) {
    options.onRequest = (r) =>
      process.stderr.write(
        `#${r.index} ${r.provider} ${r.api}${r.stream ? " (stream)" : ""} model=${r.model} "${r.lastUserMessage.slice(0, 60)}"\n`,
      );
  }

  const mock = await startMockLLM(options);
  for (const rule of script.rules) mock.on(rule.match, rule.reply, rule.times === undefined ? {} : { times: rule.times });

  process.stdout.write(
    `mock-llm-server listening on ${mock.url}\n` +
      `  OpenAI base URL:    ${mock.openaiBaseURL}\n` +
      `  Anthropic base URL: ${mock.anthropicBaseURL}\n` +
      `Press Ctrl+C to stop.\n`,
  );
  const stop = () => {
    void mock.close().then(() => process.exit(0));
  };
  process.on("SIGINT", stop);
  process.on("SIGTERM", stop);
  return -1;
}

main(process.argv.slice(2)).then(
  (code) => {
    if (code >= 0) process.exitCode = code;
  },
  (error: unknown) => {
    if (error instanceof UsageError) {
      process.stderr.write(`mock-llm-server: ${error.message}\nRun "mock-llm-server --help" for usage.\n`);
      process.exitCode = 2;
    } else if ((error as NodeJS.ErrnoException).code === "EADDRINUSE") {
      process.stderr.write(`mock-llm-server: port is already in use. Pick another with --port, or --port 0 for any free port.\n`);
      process.exitCode = 1;
    } else {
      throw error;
    }
  },
);
