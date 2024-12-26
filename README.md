
# Analyze Your Favorite Series with NLP

This project uses Natural Language Processing (NLP) and Large Language Models (LLMs) to analyze a series. It includes web scraping, text classification, character network creation, and a character chatbot, all integrated into a web GUI using Streamlit.

## Overview

### Components
1. **Web Crawler**: Scrapes data about the series (e.g., episodes, characters).
2. **Character Network**: Builds a network of character interactions using NER.
3. **Text Classifier**: Classifies text into various categories.
4. **Theme Classifier**: Extracts themes using zero-shot classification.
5. **Character Chatbot**: Chats with characters using LLMs.

### Tools Used
- Scrapy, SpaCy, NetworkX
- Scikit-learn, Hugging Face Transformers
- Streamlit for GUI

## Requirements
Install dependencies:
```bash
pip install -r requirements.txt
```

## How to Run
1. Clone the repository.
2. Install dependencies.
3. Run the app:
   ```bash
   streamlit run app.py
   ```

## Features
- Explore dataset and visualizations.
- Test classifiers and interact with the chatbot.


