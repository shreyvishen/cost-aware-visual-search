"""Web server for the cost-aware visual search page.

    python3 app/server.py            # http://localhost:3001

Stdlib only — no framework, no install step. Serves `app/static/` and one endpoint,
`POST /api/infer`, which runs a live zoom episode through `app/infer.py`.

`cgi` was removed in Python 3.13, so multipart is parsed by hand below. The form is small and
fixed (one file, a few text fields), which makes that a dozen lines rather than a project.
"""
from __future__ import annotations

import json
import mimetypes
import os
import re
import sys
import tempfile
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
STATIC = HERE / "static"
sys.path.insert(0, str(HERE.parent))

PORT = int(os.environ.get("PORT", "3001"))
VSTAR = Path.home() / "archive" / "cost-aware-vlm" / "vstar_full"
MAX_UPLOAD = 24 * 1024 * 1024

# Inference is serialised: two generations at once on one Metal GPU corrupt the timings this
# page reports as measurements.
INFER_LOCK = threading.Lock()


def parse_multipart(body: bytes, boundary: bytes) -> dict:
    """Return {field: str | (filename, bytes)} for a simple, flat multipart body."""
    out: dict = {}
    sep = b"--" + boundary
    for part in body.split(sep):
        if not part.strip() or part.strip() == b"--":
            continue
        head, _, data = part.partition(b"\r\n\r\n")
        if not _:
            continue
        data = data.rstrip(b"\r\n")
        m = re.search(rb'name="([^"]*)"', head)
        if not m:
            continue
        name = m.group(1).decode()
        fn = re.search(rb'filename="([^"]*)"', head)
        out[name] = (fn.group(1).decode(), data) if fn else data.decode("utf-8", "replace")
    return out


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # quieter than the default
        if "/api/" in (args[0] if args else ""):
            sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    # --- static ---------------------------------------------------------

    def do_GET(self):
        path = self.path.split("?")[0]
        if path.startswith("/img/"):
            return self.send_vstar(path[len("/img/"):])
        rel = "index.html" if path == "/" else path.lstrip("/")
        target = (STATIC / rel).resolve()
        if not str(target).startswith(str(STATIC.resolve())) or not target.is_file():
            return self.send_json({"error": "not found"}, 404)
        ctype = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
        body = target.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def send_vstar(self, sid: str):
        """Serve a V*Bench image by its sid, e.g. vstar-direct_attributes-sa_10033."""
        sid = sid.replace(".jpg", "")
        parts = sid.split("-")
        if len(parts) < 3 or parts[0] != "vstar" or ".." in sid or "/" in sid:
            return self.send_json({"error": "bad sid"}, 400)
        target = (VSTAR / parts[1] / (parts[2] + ".jpg")).resolve()
        if not str(target).startswith(str(VSTAR.resolve())) or not target.is_file():
            return self.send_json({"error": "image not mirrored"}, 404)
        body = target.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "public, max-age=3600")
        self.end_headers()
        self.wfile.write(body)

    # --- inference ------------------------------------------------------

    def do_POST(self):
        route = self.path.split("?")[0]
        if route == "/api/stream":
            return self.do_stream()
        if route != "/api/infer":
            return self.send_json({"error": "not found"}, 404)
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > MAX_UPLOAD:
                return self.send_json({"error": "image too large (24 MB limit)"}, 413)
            ctype = self.headers.get("Content-Type", "")
            m = re.search(r"boundary=([^;]+)", ctype)
            if not m:
                return self.send_json({"error": "expected multipart/form-data"}, 400)
            fields = parse_multipart(self.rfile.read(length), m.group(1).strip('"').encode())

            img = fields.get("image")
            question = (fields.get("question") or "").strip()
            if not isinstance(img, tuple) or not img[1]:
                return self.send_json({"error": "no image uploaded"}, 400)
            if not question:
                return self.send_json({"error": "no question"}, 400)

            model = fields.get("model", "b")
            quant = fields.get("quant", "q4")
            down = fields.get("downproject", "1") not in ("0", "false", "False")
            if model not in ("base", "a", "b") or quant not in ("q4", "q8", "bf16"):
                return self.send_json({"error": "bad model or quant"}, 400)

            suffix = Path(img[0]).suffix or ".jpg"
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
                f.write(img[1])
                tmp = f.name

            from app import infer  # imported late so a broken backend cannot block startup

            with INFER_LOCK:
                result = infer.run_episode(tmp, question, model=model, quant=quant,
                                           downproject=down)
            try:
                os.unlink(tmp)
            except OSError:
                pass
            return self.send_json(result, 200)
        except Exception as e:  # never 500 without saying why
            traceback.print_exc()
            return self.send_json({"error": f"{type(e).__name__}: {e}"}, 500)

    def do_stream(self):
        """Same inputs as /api/infer, but streams the episode as newline-delimited JSON.

        Not EventSource: that is GET-only and this needs a file upload. The client reads
        the response body incrementally with fetch + ReadableStream.
        """
        tmp = None
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > MAX_UPLOAD:
                return self.send_json({"error": "image too large (24 MB limit)"}, 413)
            m = re.search(r"boundary=([^;]+)", self.headers.get("Content-Type", ""))
            if not m:
                return self.send_json({"error": "expected multipart/form-data"}, 400)
            fields = parse_multipart(self.rfile.read(length), m.group(1).strip('"').encode())
            img = fields.get("image")
            question = (fields.get("question") or "").strip()
            if not isinstance(img, tuple) or not img[1] or not question:
                return self.send_json({"error": "need an image and a question"}, 400)
            model = fields.get("model", "b")
            quant = fields.get("quant", "q4")
            down = fields.get("downproject", "1") not in ("0", "false", "False")
            if model not in ("a", "b") or quant not in ("q4", "q8", "bf16"):
                return self.send_json({"error": "bad model or quant"}, 400)

            with tempfile.NamedTemporaryFile(suffix=Path(img[0]).suffix or ".jpg",
                                             delete=False) as f:
                f.write(img[1])
                tmp = f.name

            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson")
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Accel-Buffering", "no")
            self.send_header("Connection", "close")
            self.end_headers()

            from app import stream as streamer

            with INFER_LOCK:
                for ev in streamer.stream_episode(tmp, question, model=model, quant=quant,
                                                  downproject=down):
                    self.wfile.write((json.dumps(ev) + "\n").encode())
                    self.wfile.flush()
        except BrokenPipeError:
            pass  # the client navigated away mid-episode
        except Exception as e:
            traceback.print_exc()
            try:
                self.wfile.write((json.dumps({"type": "error",
                                              "msg": f"{type(e).__name__}: {e}"}) + "\n").encode())
                self.wfile.flush()
            except Exception:
                pass
        finally:
            if tmp:
                try:
                    os.unlink(tmp)
                except OSError:
                    pass

    # --- helper ---------------------------------------------------------

    def send_json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    if not (STATIC / "data.json").exists():
        print("data.json missing — running app/build_data.py first")
        os.system(f"{sys.executable} {HERE / 'build_data.py'}")
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    print(f"cost-aware visual search → http://localhost:{PORT}")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping")
        try:
            from app import infer
            infer.shutdown()
        except Exception:
            pass
        srv.server_close()


if __name__ == "__main__":
    main()
