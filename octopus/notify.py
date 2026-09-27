import json
import os
import urllib.request
from datetime import datetime, timezone


class Notifier:
    """Logs to console + file; important events also go to Telegram if configured
    (env TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID)."""

    def __init__(self, log_path: str | None = None, telegram: bool = False):
        self.log_path = log_path
        self.token = os.environ.get("TELEGRAM_BOT_TOKEN") if telegram else None
        self.chat = os.environ.get("TELEGRAM_CHAT_ID") if telegram else None

    def __call__(self, msg: str, important: bool = False):
        line = f"[{datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S}] {msg}"
        print(line, flush=True)
        if self.log_path:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        if important and self.token and self.chat:
            try:
                body = json.dumps({"chat_id": self.chat, "text": f"🐙 {msg}"}).encode()
                req = urllib.request.Request(f"https://api.telegram.org/bot{self.token}/sendMessage", body,
                                             {"Content-Type": "application/json"})
                urllib.request.urlopen(req, timeout=10).close()
            except Exception as e:
                print(f"telegram failed: {e}", flush=True)
