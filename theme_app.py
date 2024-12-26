import streamlit as st
import pandas as pd
from theme_classifier import ThemeClassifier
from character_network import NamedEntityRecognizer, CharacterNetworkGenerator
from text_classification import JutsuClassifier
from character_chatbot import CharacterChatBot
import os
from dotenv import load_dotenv

load_dotenv()

# Function to get themes
def get_themes(theme_list_str, subtitles_path, save_path):
    if not theme_list_str or not subtitles_path:
        st.error("Please provide themes and a valid subtitles path.")
        return None

    theme_list = [theme.strip() for theme in theme_list_str.split(',')]
    theme_classifier = ThemeClassifier(theme_list)
    output_df = theme_classifier.get_themes(subtitles_path, save_path)

    # Remove 'dialogue' from theme list
    theme_list = [theme for theme in theme_list if theme != 'dialogue']
    output_df = output_df[theme_list]

    # Create dataframe for plotting
    plot_data = pd.DataFrame({
        'Theme': theme_list,
        'Score': output_df[theme_list].sum().values
    })
    return plot_data

# Function to generate character network
def get_character_network(subtitles_path, ner_path):
    ner = NamedEntityRecognizer()
    ner_df = ner.get_ners(subtitles_path, ner_path)

    character_network_generator = CharacterNetworkGenerator()
    relationship_df = character_network_generator.generate_character_network(ner_df)
    html = character_network_generator.draw_network_graph(relationship_df)

    return html

# Function for text classification
def classify_text(text_classification_model, text_classification_data_path, text_to_classify):
    if not text_classification_model or not text_classification_data_path or not text_to_classify:
        st.error("Please provide all required inputs for text classification.")
        return None

    jutsu_classifier = JutsuClassifier(
        model_path=text_classification_model,
        data_path=text_classification_data_path,
        huggingface_token=os.getenv('huggingface_token')
    )

    output = jutsu_classifier.classify_jutsu(text_to_classify)
    return output[0]

# Function for character chatbot interaction
def chat_with_character_chatbot(message, history):
    character_chatbot = CharacterChatBot(
        "AbdullahTarek/Naruto_Llama-3-8B",
        huggingface_token=os.getenv('huggingface_token')
    )

    output = character_chatbot.chat(message, history)
    return output['content'].strip()

# Main Streamlit app
def main():
    st.title("Multi-Function Analyzer")

    # Create tabs for different functionalities
    tab1, tab2, tab3, tab4 = st.tabs([
        "🎭 Theme Classification",
        "🕸 Character Network Generator",
        "📜 Text Classification",
        "🤖 Character Chatbot"
    ])

    # Theme Classification Tab
    with tab1:
        st.header("Theme Classification (Zero-Shot Classifiers)")
        theme_list_str = st.text_input("Enter Themes", placeholder="Enter themes separated by commas")
        subtitles_path = st.text_input("Enter Subtitles or Script Path", placeholder="Enter file path")
        save_path = st.text_input("Enter Save Path", placeholder="Enter save location")

        if st.button("Get Themes", key="theme_button"):
            plot_data = get_themes(theme_list_str, subtitles_path, save_path)
            if plot_data is not None:
                st.subheader("Theme Scores")
                st.bar_chart(data=plot_data.set_index('Theme'))

    # Character Network Generator Tab
    with tab2:
        st.header("Character Network Generator")
        subtitles_path = st.text_input("Enter Subtitles or Script Path", key="network_subtitles_path")
        ner_path = st.text_input("Enter NERs Save Path", key="network_ner_path")

        if st.button("Get Character Network", key="network_button"):
            st.subheader("Character Network Graph")
            html = get_character_network(subtitles_path, ner_path)
            if html:
                st.components.v1.html(html, height=600)

    # Text Classification Tab
    with tab3:
        st.header("Text Classification with LLMs")
        text_classification_model = st.text_input("Model Path", placeholder="Enter model path")
        text_classification_data_path = st.text_input("Data Path", placeholder="Enter data path")
        text_to_classify = st.text_area("Text Input", placeholder="Enter text to classify")

        if st.button("Classify Text (Jutsu)", key="classify_text_button"):
            output = classify_text(text_classification_model, text_classification_data_path, text_to_classify)
            if output:
                st.subheader("Classification Output")
                st.write(output)

    # Character Chatbot Tab
    with tab4:
        st.header("Character Chatbot")
        history = []  # Keep track of conversation history
        user_input = st.text_input("Your Message", placeholder="Type your message here")

        if st.button("Send", key="chatbot_button"):
            if user_input:
                response = chat_with_character_chatbot(user_input, history)
                history.append((user_input, response))
                for user_msg, bot_msg in history:
                    st.write(f"**You**: {user_msg}")
                    st.write(f"**Bot**: {bot_msg}")

if __name__ == '__main__':
    main()
