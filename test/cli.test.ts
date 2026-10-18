import assert from "node:assert/strict";
import { spawn, spawnSync } from "node:child_process";
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { after, describe, it } from "node:test";

const CLI = join(import.meta.dirname, "..", "dist", "cli.js");
const dir = mkdtempSync(join(tmpdir(), "mock-llm-cli-"));
after(() => rmSync(dir, { recursive: true, force: true }));

const cli = (args: string[]) => spawnSync(process.execPath, [CLI, ...args], { encoding: "utf8", timeout: 10_000 });

describe("cli", () => {
  it("prints help and version", () => {
    assert.match(cli(["--help"]).stdout, /Usage:\n {2}mock-llm-server \[--port 8787\]/);
    const { version } = JSON.parse(readFileSync(join(import.meta.dirname, "..", "package.json"), "utf8"));
    assert.equal(cli(["-v"]).stdout.trim(), version);
  });

  it("rejects bad scripts and options with exit code 2", () => {
    const bad = join(dir, "bad.json");
    writeFileSync(bad, JSON.stringify({ replies: [{ txt: "typo" }] }));
    const r = cli(["--script", bad]);
    assert.equal(r.status, 2);
    assert.match(r.stderr, /Unknown reply field: txt/);
    assert.equal(cli(["--port", "99999"]).status, 2);
    assert.match(cli(["--script", join(dir, "missing.json")]).stderr, /can't find script/);
  });

  it("serves scripted replies", async () => {
    const script = join(dir, "ok.json");
    writeFileSync(script, JSON.stringify({ replies: ["Hello!"], rules: [{ match: "weather", reply: "Sunny" }] }));
    const child = spawn(process.execPath, [CLI, "--port", "0", "--script", script, "--quiet"]);
    try {
      const url = await new Promise<string>((resolve, reject) => {
        let out = "";
        child.stdout.on("data", (d) => {
          out += d;
          const m = /listening on (http:\/\/\S+)/.exec(out);
          if (m) resolve(m[1]!);
        });
        child.on("exit", () => reject(new Error(`exited early: ${out}`)));
      });
      const ask = async (content: string) => {
        const res = await fetch(`${url}/v1/chat/completions`, {
          method: "POST",
          body: JSON.stringify({ model: "m", messages: [{ role: "user", content }] }),
        });
        return ((await res.json()) as { choices: { message: { content: string } }[] }).choices[0]!.message.content;
      };
      assert.equal(await ask("weather?"), "Sunny");
      assert.equal(await ask("hi"), "Hello!");
    } finally {
      child.kill();
    }
  });
});
