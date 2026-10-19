import { connect, createServer } from "node:net";

/** Some VPN and firewall drivers drop loopback TCP segments written after a pause. Detect that. */
export async function loopbackDeliversLateWrites(): Promise<boolean> {
  const server = createServer((s) => {
    s.write("a");
    setTimeout(() => s.end("b"), 30);
  });
  await new Promise<void>((r) => server.listen(0, "127.0.0.1", r));
  const { port } = server.address() as { port: number };
  const c = connect(port, "127.0.0.1");
  try {
    return await new Promise<boolean>((done) => {
      let got = "";
      const timer = setTimeout(() => done(false), 1000);
      c.on("data", (d) => (got += String(d)));
      c.on("end", () => {
        clearTimeout(timer);
        done(got === "ab");
      });
      c.on("error", () => done(false));
    });
  } finally {
    c.destroy();
    server.close();
    server.unref();
  }
}
