import json
import mimetypes
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

from chatbot import T1DEducationChatbot


class ChatbotHTTPServer(ThreadingHTTPServer):
    def __init__(self, server_address, handler_class, chatbot):
        super().__init__(server_address, handler_class)
        self.chatbot = chatbot


class ChatbotRequestHandler(BaseHTTPRequestHandler):
    server_version = "T1DChatbot/1.0"

    def do_GET(self):
        parsed_path = urlparse(self.path)
        path = parsed_path.path

        if path in ("/", "/index.html"):
            self.serve_static_file("index.html")
            return

        if path.startswith("/"):
            safe_path = path.lstrip("/")
            if os.path.exists(os.path.join(ROOT_DIR, safe_path)):
                self.serve_static_file(safe_path)
                return

        self.send_json(404, {"error": "Not found"})

    def do_POST(self):
        if self.path != "/chat":
            self.send_json(404, {"error": "Not found"})
            return

        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(content_length).decode("utf-8") if content_length else "{}"
            payload = json.loads(body or "{}")
        except json.JSONDecodeError:
            self.send_json(400, {"error": "Invalid JSON payload"})
            return

        message = (payload.get("message") or "").strip()
        if not message:
            self.send_json(400, {"error": "Message is required"})
            return

        chatbot = self.server.chatbot
        intent, similarity = chatbot.find_intent(message)
        if intent is not None and similarity >= 0.70:
            reply = chatbot.get_response(intent)
        else:
            reply = chatbot.generate_fallback_response(message, intent)

        self.send_json(200, {"reply": reply})

    def log_message(self, format, *args):
        return

    def serve_static_file(self, relative_path):
        file_path = os.path.join(ROOT_DIR, relative_path)
        if not os.path.exists(file_path):
            self.send_json(404, {"error": "File not found"})
            return

        content_type = mimetypes.guess_type(file_path)[0] or "application/octet-stream"
        try:
            with open(file_path, "rb") as handle:
                content = handle.read()
        except OSError:
            self.send_json(500, {"error": "Unable to read file"})
            return

        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def send_json(self, status_code, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    chatbot = T1DEducationChatbot(os.path.join(ROOT_DIR, "intents.json"))
    server = ChatbotHTTPServer(("0.0.0.0", 8000), ChatbotRequestHandler, chatbot)
    print("Serving T1D chatbot on http://localhost:8000")
    server.serve_forever()


if __name__ == "__main__":
    main()
