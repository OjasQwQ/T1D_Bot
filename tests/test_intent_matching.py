from pathlib import Path

import pytest

from chatbot import T1DEducationChatbot


INTENTS_FILE = Path(__file__).resolve().parents[1] / "intents.json"


@pytest.fixture
def bot():
    return T1DEducationChatbot(str(INTENTS_FILE))


def test_exact_match_returns_intent(bot):
    intent, score = bot.find_intent("What is Type 1 Diabetes")
    assert intent is not None
    assert intent.get("tag") == "t1d_definition"
    assert score >= 0.7


def test_typo_tolerant_match(bot):
    # common noisy input should still resolve to the 'symptoms' intent
    intent, score = bot.find_intent("what r the symptms of t1d")
    assert intent is not None
    assert intent.get("tag") == "symptoms"
    assert score > 0.4


def test_empty_input_returns_none(bot):
    intent, score = bot.find_intent("")
    assert intent is None
    assert score == 0.0


def test_keyword_short_input_matches(bot):
    # short keyword queries should trigger keyword matching fallback
    intent, score = bot.find_intent("symptoms")
    assert intent is not None
    assert score >= 0.55
