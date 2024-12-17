import spacy
from nltk.tokenize import sent_tokenize
import pandas as pd
from ast import literal_eval
import os
import sys
import pathlib 
folder_path = pathlib.Path(__file__).parent.resolve()
sys.path.append(os.path.join(folder_path,'../'))
from utils import load_subtitles_dataset

class NamedEntityRecognizer:
    def _init_(self):
        self.nlp_model = self.load_model()

    def load_model(self):
        # Load the spaCy model correctly
        nlp = spacy.load("en_core_web_trf")
        return nlp
    
    def get_ners_interface(self, script):
        # Tokenize the script into sentences
        script_sentence = sent_tokenize(script)
        ner_output = []

        for sentence in script_sentence:
            doc = self.nlp_model(sentence)
            ners = set()
             
            # Extract PERSON entities and store their first names
            for entity in doc.ents:
                if entity.label_ == "PERSON":
                    full_name = entity.text
                    first_name = full_name.split(" ")[0].strip()
                    ners.add(first_name)
            ner_output.append(ners)

        return ner_output
    
    def get_ners(self, dataset_path, save_path=None):
        if save_path is not None and os.path.exists(save_path):
            # If a saved file exists, load it and return the DataFrame
            df = pd.read_csv(save_path)
            df['ners'] = df['ners'].apply(lambda x: literal_eval(x) if isinstance(x, str) else x) 
            return df
        
        # Load the dataset
        df = load_subtitles_dataset(dataset_path)
        
        # Check if 'script' column exists
        if 'script' not in df.columns:
            raise ValueError("The dataset must contain a 'script' column.")
        
        df = df.head(10)  # For testing, limit to the first 10 rows

        # Run inference on the 'script' column
        df['ners'] = df['script'].apply(self.get_ners_interface)

        # Save the result if save_path is provided
        if save_path is not None:
            df.to_csv(save_path,index=False)