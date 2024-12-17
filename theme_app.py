import streamlit as st
import pandas as pd
from theme_classifier import ThemeClassifier

def get_themes(theme_list_str, subtitles_path, save_path):
    # Input validation
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

def main():
    st.title("Theme Classification (Zero-Shot Classifiers)")

    # Sidebar for inputs
    st.sidebar.header("Input Parameters")
    theme_list_str = st.sidebar.text_input("Themes", placeholder="Enter themes separated by commas")
    subtitles_path = st.sidebar.text_input("Subtitles or script Path", placeholder="Enter file path")
    save_path = st.sidebar.text_input("Save Path", placeholder="Enter save location")

    # Button to trigger the classification
    if st.sidebar.button("Get Themes"):
        # Get themes and display plot
        plot_data = get_themes(theme_list_str, subtitles_path, save_path)
        
        if plot_data is not None:
            st.subheader("Theme Scores")
            st.bar_chart(data=plot_data.set_index('Theme'))  # Use Streamlit's bar chart function

if __name__ == '__main__':
    main()