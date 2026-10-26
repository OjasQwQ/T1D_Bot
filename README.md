# T1D Education Chatbot

A rule-based chatbot designed to educate people about Type 1 Diabetes (T1D) and spread awareness. It matches user queries against locally stored intents and responds from the built-in knowledge base.

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
3. **Fallback Handling**: Uses local keyword matching and guidance when confidence in an intent match is low. The chatbot does not use AI models, external model services, or API keys.

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

## Dependencies

The chatbot runtime uses only the Python standard library. It does not load AI models, make external model API calls, or require API keys.

For development tests, install pytest and run:

```bash
pip install pytest
python -m pytest -q
```

A lightweight test runner is also available at `tests/run_intent_tests.py` and requires no third-party packages.
