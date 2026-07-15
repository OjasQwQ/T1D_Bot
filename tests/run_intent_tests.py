from pathlib import Path
import sys
import os

# Ensure project root is on sys.path so we can import chatbot when running this script directly
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from chatbot import T1DEducationChatbot


INTENTS_FILE = Path(__file__).resolve().parents[1] / "intents.json"


def fail(msg: str):
    print("FAIL:", msg)
    sys.exit(1)


def main():
    bot = T1DEducationChatbot(str(INTENTS_FILE))

    intent, score = bot.find_intent("What is Type 1 Diabetes")
    if intent is None or intent.get("tag") != "t1d_definition" or score < 0.7:
        fail("Exact match test failed")

    intent, score = bot.find_intent("what r the symptms of t1d")
    if intent is None or intent.get("tag") != "symptoms" or score <= 0.4:
        fail("Typo-tolerant match test failed")

    intent, score = bot.find_intent("")
    if intent is not None or score != 0.0:
        fail("Empty input test failed")

    intent, score = bot.find_intent("symptoms")
    if intent is None or score < 0.55:
        fail("Keyword short input test failed")

    print("All intent-matching tests passed.")


if __name__ == "__main__":
    main()
