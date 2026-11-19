"""BahuSanket live front end on this computer: python server.py, then open http://localhost:8000"""
import http.server, json, os, socketserver, sys, threading, time, traceback, webbrowser

HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")       # any operation the Apple GPU lacks runs on the CPU instead
os.environ.setdefault("HF_HUB_OFFLINE", "1")                    # everything was downloaded by setup.sh: no internet needed
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import pipeline as P

PORT = int(os.environ.get("BAHU_PORT", 8000))
WEB = os.path.join(HERE, "web")
FILES = {"/": os.path.join(WEB, "app.html"), "/app.html": os.path.join(WEB, "app.html"), "/viewer.html": os.path.join(WEB, "viewer.html"),
         "/avatar.b64.txt": os.path.join(P.BUNDLE, "avatar.b64.txt"),
         "/clips_paragraph.json": os.path.join(P.BUNDLE, "clips_paragraph_19.json"),
         "/clips_model.json": os.path.join(P.BUNDLE, "clips_model_18.json"),
         "/clips.json": os.path.join(P.BUNDLE, "clips.json")}
TYPES = {".html": "text/html; charset=utf-8", ".json": "application/json", ".txt": "text/plain"}
SIGNER = None


class Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype):
        self.send_response(code); self.send_header("Content-Type", ctype); self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store"); self.end_headers(); self.wfile.write(body)

    def _json(self, obj, code=200):
        self._send(code, json.dumps(obj, ensure_ascii=False, separators=(",", ":")).encode("utf-8"), "application/json")

    def do_GET(self):
        p = self.path.split("?")[0]
        if p == "/api/health":
            name = {"mps": "Apple GPU", "cuda": "NVIDIA GPU", "cpu": "CPU"}[P.DEVICE]
            return self._json({"ok": SIGNER is not None, "gpu": f"this laptop ({name})", "langs": list(P.LANGS)})
        f = FILES.get(p)
        if p == "/clips.json" and not os.path.exists(f):
            return self._send(200, b"[]", "application/json")
        if not f or not os.path.exists(f):
            return self._send(404, b"not found", "text/plain")
        self.send_response(200); self.send_header("Content-Type", TYPES.get(os.path.splitext(f)[1], "application/octet-stream"))
        self.send_header("Content-Length", str(os.path.getsize(f))); self.end_headers()
        with open(f, "rb") as fh:
            while True:
                b = fh.read(1 << 20)
                if not b:
                    break
                self.wfile.write(b)

    def do_POST(self):
        if self.path.split("?")[0] != "/api/generate":
            return self._send(404, b"not found", "text/plain")
        try:
            req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            if SIGNER is None:
                return self._json({"ok": False, "error": "the models are still loading, try again in a moment"})
            self._json(SIGNER.make_paragraph(req.get("text", ""), req.get("lang", "auto")))
        except Exception as e:
            traceback.print_exc()
            self._json({"ok": False, "error": f"{type(e).__name__}: {e}"})


class Server(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def main():
    global SIGNER
    srv = Server(("127.0.0.1", PORT), Handler)          # only this computer can open it
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    log = lambda m: print(m, flush=True)
    log(f"Loading the models (about a minute) ... device: {P.DEVICE}")
    s = P.Signer(log)
    if os.environ.get("BAHU_SKIP_WARMUP") != "1":
        t0 = time.time()
        s.make_paragraph("Thank you for coming today.", "en")         # first run of each step is the slowest
        s.make_paragraph("आप सभी का स्वागत है।", "hi")                    # also loads IndicTrans2
        s.seen.clear(); s.n = 0
        log(f"warm-up done [{time.time() - t0:.0f}s]")
    SIGNER = s
    url = f"http://localhost:{PORT}/"
    log(f"\nREADY: open {url}   (Ctrl+C here stops it)\n")
    if os.environ.get("BAHU_NO_BROWSER") != "1":
        webbrowser.open(url)
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        log("stopped")


if __name__ == "__main__":
    main()
