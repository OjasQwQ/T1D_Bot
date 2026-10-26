import difflib
import json
import os
import random
import re
import string
from typing import Dict, List, Optional


class T1DEducationChatbot:
    def __init__(self, intents_file: str):
        """Initialize the chatbot with the built-in intent knowledge base."""
        self.intents = self.load_intents(intents_file)
        self.context = {}
        self.pattern_list: List[Dict] = []
        self._index_intents()

    def _index_intents(self) -> None:
        """Build a normalized list of patterns for intent matching."""
        for intent in self.intents:
            patterns = intent.get("patterns", [])
            for pattern in patterns:
                self.pattern_list.append({
                    "intent": intent,
                    "pattern": self.normalize_text(pattern),
                })

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

    def generate_fallback_response(self, user_input: str, intent: Optional[Dict] = None) -> str:
        """Return a safe local fallback response for unmatched input."""
        violation = self._detect_disallowed_content(user_input)
        if violation:
            if violation == "self-harm":
                return (
                    "If you are thinking about harming yourself, please seek immediate help. "
                    "Contact your local emergency number right now. "
                    "If you are in the United States, call or text 988 (Suicide & Crisis Lifeline). "
                    "In the UK, call Samaritans at 116 123. In India, you can contact AASRA at +91 22 2754 6669 or iCall at +91 9152987821. "
                    "For other countries, local emergency services or your national suicide prevention hotline can help. "
                    "If you are able, tell someone you trust you are in danger."
                )
            return (
                "I'm sorry, I can't assist with requests that are violent, sexual, or illegal. "
                "If you're in danger, please contact local emergency services."
            )

        if intent is not None:
            return self.get_response(intent)

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

    def _detect_disallowed_content(self, text: str) -> Optional[str]:
        """Detect disallowed or harmful content categories in user input.

        Returns the category string when a match is found, otherwise `None`.
        """
        normalized = self.normalize_text(text)

        disallowed_patterns = {
            "sexual": [r"\bsex\b", r"porn", r"sexual", r"sexually", r"\bincest\b"],
            "self-harm": [r"suicid", r"kill myself", r"harm myself", r"cut myself", r"want to die"],
            "violence": [r"how to kill", r"murder", r"assassin", r"bomb", r"explosive", r"shoot people", r"attack"],
            "illegal": [r"how to make meth", r"how to make a bomb", r"illegal", r"hack into", r"\bhack\b", r"steal", r"forge", r"\bmeth\b", r"cocaine", r"heroin", r"\bdrug\b dealer", r"sell drugs", r"buy cocaine", r"buy heroin", r"fentanyl"],
        }

        for category, patterns in disallowed_patterns.items():
            for pattern in patterns:
                try:
                    if re.search(pattern, normalized):
                        return category
                except re.error:
                    # ignore bad regex patterns
                    continue

        return None

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
                if intent is not None and similarity >= 0.65:
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