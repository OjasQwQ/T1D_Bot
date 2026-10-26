# T1D Education Chatbot

A hybrid (rule based and learning based) chatbot designed to educate people about Type 1 Diabetes (T1D) and spread awareness. The chatbot uses NLP techniques to understand user queries and provide accurate, helpful information about T1D.

I made this chatbot for a youth led NGO website which I am a part of. The NGO **DiAware** is working towards spreading awareness about Diabetes, especially T1D.

## Features/Uses

- **Educational Content**: Comprehensive information about T1D including definition, symptoms, treatment, complications, and more
- **Myth Busting**: Addresses common misconceptions about Type 1 Diabetes
- **Emergency Info**: Provides guidance on recognizing and responding to diabetes emergencies
- **Resource Direction**: Points users to reputable T1D organizations and resources
- **Research Updates**: Info about current research and potential future treatments
- **Lifestyle Guidance**: Advice on living well with T1D including exercise, nutrition, and daily management

## How It Works

1. **Intent Classification**: The chatbot analyzes user input to determine the intent using similarity matching from the intents.json file which also acts as constraints for the bot.
2. **Response Selection**: Provides appropriate responses from the matched intent
3. **Fallback Handling**: Uses google genai to respnd to queries when confidence level is low against preset intents.

## File Structure

- `intents.json`: Contains the knowledge base with intents, patterns, and responses
- `chatbot.py`: Main chatbot implementation with NLP processing and conversation logic
- `README.md`: This documentation file!

## Knowledge Base Structure (intents.json)

The knowledge base is organized into intents, each containing:
- **tag**: Unique identifier for the intent
- **patterns**: Example user inputs that would match this intent
- **responses**: Possible bot responses for this intent

## Setup and Usage

### Prerequisites
- Python 3.x

### Installation
1. Clone or download this repository
2. Ensure you have Python 3.x installed
3. No additional packages are required (uses only standard library)

### Running the Chatbot
```bash
python chatbot.py
```

### Conversation Examples
- User: "What is T1D?"
- User: "What are the symptoms of Type 1 Diabetes?"
- User: "How is T1D different from Type 2?"
- User: "What should I do if someone has low blood sugar?"

## Educational Topics Covered

1. **T1D Definition and Basics**
2. **T1D vs T2D Differences**
3. **Symptoms and Warning Signs**
4. **Diagnosis Process**
5. **Treatment and Management**
6. **Complications**
7. **Hypoglycemia and Hyperglycemia**
8. **Diabetic Ketoacidosis (DKA)**
9. **Common Myths and Misconceptions**
10. **Living with T1D**
11. **Emergency Procedures**
12. **Resources and Support**
13. **Current Research**

## Design Principles

- **Accuracy**: Information based on current medical understanding
- **Accessibility**: Language appropriate for general audiences
- **Empathy**: Respectful and supportive tone
- **Actionability**: Provides practical information users can apply
- **Comprehensiveness**: Covers key aspects of T1D education and awareness

## Disclaimer

This chatbot provides educational information only and is not a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition.

## Contributing

Feel free to submit issues or pull requests to improve the chatbot's knowledge base or functionality.

## License

This project is open source and available for educational and awareness purposes.

## Dependencies & Libraries

This project is intentionally lightweight and uses the Python standard library for core functionality. A few optional third-party libraries enable extended features (LLM fallback, tests, and local model inference):

- **Python Standard Library**: `difflib`, `json`, `os`, `random`, `re`, `string`, and `typing` are used for intent matching, normalization, I/O, and basic program flow. No installation is required for these.
- **google-genai** (optional): Official Gemini (Google) client used when `LLM_BACKEND=gemini` and `GEMINI_API_KEY` is provided. Enables `models.generate_content` and `interactions.create` calls for LLM fallback answers. Install with `pip install google-genai` and set `GEMINI_API_KEY` in `.env`.
- **transformers** (optional): Used only if you enable the `nemotron-3b` backend (local or remote inference) to run the `nvidia/Nemotron-3B-Instruct` pipeline as a fallback. Install with `pip install transformers` and follow model-specific instructions for runtime (may require `torch` or other acceleration libraries).
- **torch / accelerate** (optional): Backend runtime for `transformers` models. Install per PyTorch platform instructions (GPU/CPU specific). Example: `pip install torch` or see https://pytorch.org for correct wheel.
- **pytest** (development): Used to run unit tests in `tests/`. Install with `pip install pytest` and run `python -m pytest -q`. A lightweight test runner is also provided at `tests/run_intent_tests.py` so you can verify tests without installing `pytest`.

Installation examples (optional dependencies):

```bash
# Install Gemini client and testing tools
pip install google-genai pytest

# Optional: install transformers + torch for Nemotron fallback (may require GPU or specific wheels)
pip install transformers torch
```

Notes:
- The LLM fallback features are optional — the bot works offline using the `intents.json` knowledge base and the built-in matching logic.
- Model availability and API access depend on your account and plan for cloud LLMs; programmatic model discovery or switching may be necessary if a model ID is unavailable.
