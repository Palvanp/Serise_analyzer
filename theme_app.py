import streamlit as st
import pandas as pd
from theme_classifier import ThemeClassifier
from character_network_generator import NamedEntityRecognizer, CharacterNetworkGenerator

# Function to get themes
def get_themes(theme_list_str, subtitles_path, save_path):
    if not theme_list_str or not subtitles_path:
        st.error("Please provide themes and a valid subtitles path.")
        return None
    
    theme_list = [theme.strip() for theme in theme_list_str.split(',')]
    theme_classifier = ThemeClassifier(theme_list)
    output_df = theme_classifier.get_themes(subtitles_path, save_path)

    # Remove dialogue and process themes
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

# Main Streamlit app
def main():
    st.title("Theme and Character Network Analyzer")

    # Create tabs for Theme Classifier and Character Network Generator
    tab1, tab2 = st.tabs(["🎭 Theme Classifier", "🕸 Character Network Generator"])

    # Theme Classifier Section
    with tab1:
        st.header("Theme Classification (Zero-Shot Classifiers)")
        theme_list_str = st.text_input("Enter Themes", placeholder="Enter themes separated by commas")
        subtitles_path = st.text_input("Enter Subtitles or Script Path", placeholder="Enter file path")
        save_path = st.text_input("Enter Save Path", placeholder="Enter save location")

        # Button to trigger theme classification
        if st.button("Get Themes"):
            plot_data = get_themes(theme_list_str, subtitles_path, save_path)
            if plot_data is not None:
                st.subheader("Theme Scores")
                st.bar_chart(data=plot_data.set_index('Theme'))

    # Character Network Generator Section
    with tab2:
        st.header("Character Network Generator")
        subtitles_path = st.text_input("Enter Subtitles or Script Path", key="network_subtitles_path")
        ner_path = st.text_input("Enter NERs Save Path", key="network_ner_path")

        # Button to trigger character network generation
        if st.button("Get Character Network"):
            st.subheader("Character Network Graph")
            html = get_character_network(subtitles_path, ner_path)
            if html:
                st.components.v1.html(html, height=600)

if __name__ == '__main__':
    main()