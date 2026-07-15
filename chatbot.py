import difflib
import json
import os
import random
import re
import string
from typing import Dict, List, Optional


def load_environment_variables() -> None:
    """Load values from a local .env file when present."""
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if not os.path.exists(env_path):
        return

    with open(env_path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


load_environment_variables()

try:
    from google import genai as google_genai
except ImportError:  # pragma: no cover - optional dependency guard
    google_genai = None


class T1DEducationChatbot:
    def __init__(self, intents_file: str):
        """Initialize the chatbot using sentence embeddings and an optional LLM fallback."""
        self.intents = self.load_intents(intents_file)
        self.context = {}
        self.pattern_list: List[Dict] = []
        self.knowledge_base: List[str] = []
        self.llm_backend = os.getenv("LLM_BACKEND", "gemini").lower()
        self.llm_model = os.getenv("LLM_MODEL", "gemini-2.5-flash")
        self.llm_client = None
        self.nemotron_pipeline = None

        self._index_intents()
        self._init_llm_client()

    def _index_intents(self) -> None:
        """Build a normalized pattern list and the knowledge-base strings."""
        for intent in self.intents:
            tag = intent.get("tag", "unknown")
            patterns = intent.get("patterns", [])
            responses = intent.get("responses", [])
            self.knowledge_base.append(
                f"Topic: {tag}\n"
                f"Questions: {', '.join(patterns)}\n"
                f"Answers: {' '.join(responses)}"
            )
            for pattern in patterns:
                self.pattern_list.append({
                    "intent": intent,
                    "pattern": self.normalize_text(pattern),
                })

    def _init_llm_client(self) -> None:
        """Initialize Gemini if the API key is available."""
        if self.llm_backend != "gemini" or google_genai is None:
            return

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            return

        try:
            self.llm_client = google_genai.Client(api_key=api_key)
        except Exception as exc:  # pragma: no cover - runtime guard
            print(f"[WARN] Gemini client initialization failed: {exc}")
            self.llm_client = None

    @staticmethod
    def load_intents(file_path: str) -> List[Dict]:
        """Load intents from a JSON file."""
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                data = json.load(file)
            return data.get("intents", [])
        except FileNotFoundError:
            print(f"[ERROR] Could not find '{file_path}'.")
        except json.JSONDecodeError as e:
            print(f"[ERROR] Invalid JSON: {e}")

        return []

    def normalize_text(self, text: str) -> str:
        """Normalize text for more forgiving matching."""
        text = text.lower()
        text = text.translate(str.maketrans("", "", string.punctuation))
        replacements = {
            r"\babt\b": "about",
            r"\bplz\b": "please",
            r"\bwht\b": "what",
            r"\br\b": "are",
            r"\bu\b": "you",
            r"\bim\b": "i am",
            r"\bdont\b": "do not",
            r"\bdoesnt\b": "does not",
            r"\bcan't\b": "cannot",
            r"\bwon't\b": "will not",
        }
        for pattern, replacement in replacements.items():
            text = re.sub(pattern, replacement, text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def preprocess_text(self, text: str) -> List[str]:
        """Preprocess text: lowercase, remove punctuation, tokenize."""
        normalized = self.normalize_text(text)
        tokens = normalized.split()
        return [token.strip() for token in tokens if token.strip()]

    def calculate_similarity(self, tokens1: List[str], tokens2: List[str]) -> float:
        """Calculate similarity between two token lists using Jaccard similarity."""
        if not tokens1 or not tokens2:
            return 0.0

        set1 = set(tokens1)
        set2 = set(tokens2)
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))

        if union == 0:
            return 0.0

        return intersection / union

    def normalize_text(self, text: str) -> str:
        """Normalize text for more forgiving matching."""
        cleaned = text.lower().strip()
        cleaned = cleaned.translate(str.maketrans("", "", string.punctuation))
        cleaned = re.sub(r"\s+", " ", cleaned)
        return cleaned

    def find_intent(self, user_input: str):
        """Find the most similar intent using fuzzy pattern matching."""
        normalized_input = self.normalize_text(user_input)
        if not normalized_input:
            return None, 0.0

        best_intent = None
        best_score = 0.0
        input_tokens = set(normalized_input.split())

        for item in self.pattern_list:
            pattern = item["pattern"]
            pattern_tokens = set(pattern.split())
            ratio = difflib.SequenceMatcher(None, normalized_input, pattern).ratio()
            overlap = self.calculate_similarity(list(input_tokens), list(pattern_tokens))
            score = max(ratio, overlap * 1.25)

            if score > best_score:
                best_score = score
                best_intent = item["intent"]

        if best_score < 0.45:
            keyword_matches = []
            for intent in self.intents:
                for pattern in intent.get("patterns", []):
                    normalized_pattern = self.normalize_text(pattern)
                    if normalized_pattern in normalized_input or normalized_pattern.replace(" ", "") in normalized_input.replace(" ", ""):
                        keyword_matches.append(intent)
                        break

            if keyword_matches:
                return keyword_matches[0], 0.55

            return None, best_score

        return best_intent, best_score

    def get_response(self, intent: Dict) -> str:
        """Get a random response from the matched intent."""
        responses = intent.get("responses", [])
        if responses:
            return random.choice(responses)
        return "I'm not sure how to respond to that. Could you rephrase your question?"

    def build_llm_prompt(self, user_input: str) -> str:
        """Create a prompt that uses the intents JSON as the knowledge base."""
        knowledge_context = "\n\n".join(self.knowledge_base)
        return f"""You are Dia, a helpful Type 1 Diabetes education assistant.
Use ONLY the knowledge below when answering.

{knowledge_context}

If the question is unrelated to Type 1 Diabetes, politely say so and redirect back to T1D topics.
Keep the answer concise and under 120 words.

User: {user_input}
Assistant:"""

    def generate_fallback_response(self, user_input: str, intent: Optional[Dict] = None) -> str:
        """Try Gemini first, then Nemotron if configured, and finally fall back to a knowledge-based reply."""
        if intent is not None:
            return self.get_response(intent)

        if self.llm_backend == "gemini" and self.llm_client is not None:
            try:
                prompt = self.build_llm_prompt(user_input)
                response = self.llm_client.models.generate_content(
                    model=self.llm_model,
                    contents=prompt,
                )
                text = getattr(response, "text", None)
                if text:
                    return text.strip()
            except Exception as exc:  # pragma: no cover - runtime guard
                print(f"[WARN] Gemini fallback failed: {exc}")

        if intent is not None:
            return self.get_response(intent)

        if self.llm_backend == "nemotron-3b":
            try:
                from transformers import pipeline

                if self.nemotron_pipeline is None:
                    self.nemotron_pipeline = pipeline(
                        "text-generation",
                        model="nvidia/Nemotron-3B-Instruct",
                        trust_remote_code=True,
                    )

                prompt = self.build_llm_prompt(user_input)
                output = self.nemotron_pipeline(prompt, max_new_tokens=180, do_sample=False)
                generated = output[0]["generated_text"]
                if generated:
                    return generated.split("Assistant:", 1)[-1].strip()
            except Exception as exc:  # pragma: no cover - runtime guard
                print(f"[WARN] Nemotron fallback failed: {exc}")

        keyword_topic = self._infer_topic_from_text(user_input)
        if keyword_topic:
            return (
                f"I can help with {keyword_topic}. "
                "Ask me about symptoms, treatment, emergencies, or daily management."
            )

        return (
            "I can help with T1D topics such as definition, symptoms, treatment, emergencies, and daily management. "
            "Try asking in a simpler way, like 'what is t1d' or 'symptoms of t1d'."
        )

    def _infer_topic_from_text(self, user_input: str) -> str:
        """Infer a likely topic from keywords when the matching is weak."""
        normalized = self.normalize_text(user_input)
        topic_map = {
            "definition": ["what is t1d", "what is type 1", "define", "meaning"],
            "symptoms": ["symptom", "sign", "warning"],
            "treatment": ["treat", "treatment", "manage", "management", "care"],
            "complications": ["complication", "risk", "danger", "long term"],
            "hypoglycemia": ["low blood sugar", "hypo", "low sugar"],
            "hyperglycemia": ["high blood sugar", "hyper"],
            "emergency": ["emergency", "hospital", "911", "er"],
            "lifestyle": ["exercise", "diet", "live", "daily", "travel"],
            "myths": ["myth", "misconception", "false", "wrong"],
            "resources": ["resource", "support", "organization", "community"],
        }

        for topic, keywords in topic_map.items():
            if any(keyword in normalized for keyword in keywords):
                return topic

        return ""

    def chat(self):
        """Main chat loop."""
        print("=" * 50)
        print("T1D Education Chatbot")
        print("Learn about Type 1 Diabetes and spread awareness!")
        print("Type 'quit', 'exit', or 'bye' to end the conversation.")
        print("=" * 50)

        while True:
            try:
                user_input = input("\nYou: ").strip()

                if user_input.lower() in ["quit", "exit", "bye", "goodbye"]:
                    print("\nChatbot: Thank you for learning about T1D today! Remember, knowledge saves lives. Goodbye! 👋")
                    break

                if not user_input:
                    print("Chatbot: Please ask a question about Type 1 Diabetes.")
                    continue

                intent, similarity = self.find_intent(user_input)
                if intent is not None and similarity >= 0.70:
                    response = self.get_response(intent)
                else:
                    response = self.generate_fallback_response(user_input, intent)

                try:
                    print(f"\nChatbot: {response}")
                except UnicodeEncodeError:
                    clean_response = "".join(char for char in response if ord(char) < 65536)
                    print(f"\nChatbot: {clean_response}")

            except EOFError:
                print("\nChatbot: No more input available. Thank you for learning about T1D today! Goodbye.")
                break
            except KeyboardInterrupt:
                print("\n\nChatbot: Session interrupted. Remember to stay informed about T1D! Goodbye!")
                break
            except Exception as e:
                print(f"\nChatbot: An error occurred: {e}")
                print("Please try again or restart the conversation.")


def main():
    """Main function to run the chatbot."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    intents_file = os.path.join(script_dir, "intents.json")

    chatbot = T1DEducationChatbot(intents_file)

    if not chatbot.intents:
        print("Failed to load intents. Please check the intents.json file.")
        return

    chatbot.chat()


if __name__ == "__main__":
    main()