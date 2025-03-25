import re
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import os
import pandas as pd
import numpy as np
import time
from glob import glob
import threading  # To run preprocessing in a separate thread
import tensorflow as tf
from gensim.models import Word2Vec
from keras.saving.save import load_model
from textblob import TextBlob
from termcolor import colored
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()
vectorizer = TfidfVectorizer(max_features=50)
# autoencoder = load_model('autoencoder.h5')
def extract_contract_with_events(solidity_code):
    # Regular expression to match a contract and everything inside the braces
    # This pattern will match the first contract with all its content (including events)
    pattern = r'\bcontract\s+([A-Za-z_][A-Za-z0-9_]*)\s*\{(.*?)\}'

    # Search for the first contract with its content
    match = re.search(pattern, solidity_code, re.DOTALL)

    if match:
        contract_name = match.group(1)  # Capture contract name
        contract_content = match.group(2)  # Capture everything inside the curly braces
        return contract_name, contract_content
    else:
        return None, None  # No contract found

def replace_symbols(text):
    # Remove newlines
    text = text.replace('\n', ' ')
    # Replace more than two spaces with ' more_than_two_spaces '
    text = re.sub(r' {2,}', ' spaces ', text)
    # Replace specific symbols with their text representation
    replacements = {
        ':': 'colon',
        ';': 'semicolon',
        ',': 'comma',
        '(': 'open_parenthesis',
        ')': 'close_parenthesis',
        '{': 'open_brace',
        '}': 'close_brace',
        '==': 'equal_to',
        '&&': 'and',
        '||': 'or',
        '!': 'not',
        '+': 'add',
        '-': 'subtraction',
        '*': 'multiply',
        '/': 'divided_by',
        '=': 'equal',
        '>': 'greater_than',
        '<': 'lesser_than',

    }
    for symbol, replacement in replacements.items():
        text = text.replace(symbol, replacement)
    return text


def Feature_extraction(text):
    # lemmatization
    lemmatized_text = ' '.join([lemmatizer.lemmatize(w) for w in TextBlob(text).words])
    # Tokenization using TextBlob's word method
    # tokenized_words = TextBlob(lemmatized_text).words
    """
    Feature Extraction
    """
    print(colored(" Feature Extraction", color='blue', on_color='on_grey'))
    # Fit the vectorizer to the sentence and transform the sentence to TF-IDF vectors
    matrix = vectorizer.fit_transform([lemmatized_text])
    feat_1 = matrix.data
    l = 50 - feat_1.shape[0]
    if l == 0:
        feat1 = feat_1
    else:
        x = np.zeros(l)
        feat1 = np.hstack((feat_1, x))
    # autoencoder_graph feature
    y = feat1.reshape(1, feat1.shape[0])
    # feat3 = autoencoder.predict(y)
    # feat3 = feat3.reshape(feat3.shape[1] * feat3.shape[0])

    # word2vec
    words = lemmatized_text.split()
    max_voc = 20
    model = Word2Vec(words, max_final_vocab=max_voc, vector_size=50, window=5, min_count=1, workers=4)
    ft1 = model.cum_table  # cumulative-distribution table
    ft2 = model.syn1neg  # all feature for vocab
    l = max_voc - ft1.shape[0]
    x = np.zeros(l)
    feat2 = np.hstack((ft1, x))
    # stacking all features
    # features = np.hstack((feat1, feat3, feat2))
    features = np.hstack((feat1, feat2, feat1))
    return features


class SolFileInterface111:
    def __init__(self, root):
        self.root = root
        self.root.title("Solidity File Interface")

        # Add file selection button
        self.select_file_button = tk.Button(self.root, text="Select .sol file", command=self.select_file)
        self.select_file_button.pack(pady=20)

        # Green tick checkbox (Initially unchecked)
        self.green_tick_var = tk.BooleanVar(value=False)
        self.green_tick_checkbox = tk.Checkbutton(self.root, text="File Selected", variable=self.green_tick_var,
                                                  state="disabled", fg="green")
        self.green_tick_checkbox.pack(pady=10)

        # Progress bar for preprocessing (loader)
        self.progress = ttk.Progressbar(self.root, length=100, mode='indeterminate')
        self.progress.pack(pady=20)

        # Button for preprocessing
        self.preprocess_button = tk.Button(self.root, text="Preprocess", state="disabled", command=self.preprocess_file)
        self.preprocess_button.pack(pady=20)

        # Prediction button (will show vulnerability or not)
        self.prediction_button = tk.Button(self.root, text="Predict", state="disabled", command=self.make_prediction)
        self.prediction_button.pack(pady=10)

        # Refresh button
        self.refresh_button = tk.Button(self.root, text="Refresh", command=self.refresh_interface)
        self.refresh_button.pack(pady=10)

        # Exit button
        self.exit_button = tk.Button(self.root, text="Exit", command=self.exit_interface)
        self.exit_button.pack(pady=10)

        self.file_path = ""

    def select_file(self):
        # Open file dialog to select .sol file
        file_path = filedialog.askopenfilename(
            initialdir="Etherium Smart Contract/Etherium Smart Contract/smart contracts/block number dependency (BN)",
            filetypes=[("Solidity Files", "*.sol")])

        if file_path:
            self.file_path = file_path
            # Enable the green tick box and show that the file is selected
            self.green_tick_var.set(True)
            self.green_tick_checkbox.config(state="normal")
            # Enable the preprocess button
            self.preprocess_button.config(state="normal")

    def preprocess_file(self):
        threading.Thread(target=self.run_preprocessing).start()

    def Preprocess(self):
        with open(self.file_path, 'r', encoding='utf8', errors='ignore') as file:
            solidity_code = file.read()
        contract_name, contract_content = extract_contract_with_events(solidity_code)
        # preprocess
        out = replace_symbols(contract_content)
        # feature extraction
        feat = Feature_extraction(out)
        colored(" Features Extracted >>>>>>>>", color='blue', on_color='on_grey')
        return feat

    def run_preprocessing(self):
        # Show loader (start progress bar)
        self.progress.start()
        # Run the actual preprocessing function (this could be slow, hence running in a thread)
        try:
            self.feat = self.Preprocess()
            self.feat = np.expand_dims(self.feat, axis=-1)
            self.feat = np.expand_dims(self.feat, axis=0)
            # When preprocessing is done, stop the progress bar and show result
            self.progress.stop()
            # Enable prediction button
            self.prediction_button.config(state="normal")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred during preprocessing: {e}")
            self.progress.stop()

    def make_prediction(self):
        # Load the pre-trained model (assuming it's saved in 'proposed_model.h5')
        model = tf.keras.models.load_model('proposed_model.h5')
        out = np.argmax(model.predict(self.feat), axis=1)
        # Show prediction result
        if out == 0:
            self.prediction_button.config(text="No Vulnerability", fg="green")
        elif out == 1:
            self.prediction_button.config(text="Vulnerability Detected", fg="red")

    def refresh_interface(self):
        # Reset all UI components and clear file selection
        self.green_tick_var.set(False)
        self.green_tick_checkbox.config(state="disabled")
        self.file_path = ""
        self.progress.stop()
        self.prediction_button.config(state="disabled", text="Predict", fg="black")

    def exit_interface(self):
        # Close the interface
        self.root.quit()

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import threading
import numpy as np
import tensorflow as tf


class SolFileInterface11:
    def __init__(self, root):
        self.root = root
        self.root.title("Solidity File Interface")

        # Set window size and background color
        self.root.geometry("500x500")
        self.root.config(bg='#88a2c4')

        # Add file selection button
        self.select_file_button = tk.Button(self.root, text="Select .sol file", command=self.select_file,
                                            bg="#80558c", fg="white", font=("Arial", 12))
        self.select_file_button.pack(pady=20)

        # Green tick checkbox (Initially unchecked)
        self.green_tick_var = tk.BooleanVar(value=False)
        self.green_tick_checkbox = tk.Checkbutton(self.root, text="File Selected", variable=self.green_tick_var,
                                                  state="disabled", fg="white", bg="black", font=("Arial", 10))
        self.green_tick_checkbox.pack(pady=10)

        # Progress bar for preprocessing (loader)
        self.progress = ttk.Progressbar(self.root, length=100, mode='indeterminate')
        self.progress.pack(pady=20)

        # Button for preprocessing
        self.preprocess_button = tk.Button(self.root, text="Preprocess", state="disabled", command=self.preprocess_file,
                                           bg="#80558c", fg="black", font=("Arial", 12))
        self.preprocess_button.pack(pady=20)

        # Prediction button (will show vulnerability or not)
        self.prediction_button = tk.Button(self.root, text="Predict", state="disabled", command=self.make_prediction,
                                           bg="#80558c", fg="black", font=("Arial", 12))
        self.prediction_button.pack(pady=10)

        # Refresh button
        self.refresh_button = tk.Button(self.root, text="Refresh", command=self.refresh_interface,
                                        bg="#f44336", fg="white", font=("Arial", 12))
        self.refresh_button.pack(pady=10)

        # Exit button
        self.exit_button = tk.Button(self.root, text="Exit", command=self.exit_interface, bg="#f44336", fg="white",
                                      font=("Arial", 12))
        self.exit_button.pack(pady=10)

        self.file_path = ""

    def select_file(self):
        # Open file dialog to select .sol file
        file_path = filedialog.askopenfilename(
            initialdir="Etherium Smart Contract/Etherium Smart Contract/smart contracts/block number dependency (BN)",
            filetypes=[("Solidity Files", "*.sol")])

        if file_path:
            self.file_path = file_path
            # Enable the green tick box and show that the file is selected
            self.green_tick_var.set(True)
            self.green_tick_checkbox.config(state="normal", fg="green", text="File Selected")
            # Enable the preprocess button
            self.preprocess_button.config(state="normal", bg="#80558c", fg="white", text="Preprocess")

    def preprocess_file(self):
        threading.Thread(target=self.run_preprocessing).start()

    def Preprocess(self):
        with open(self.file_path, 'r', encoding='utf8', errors='ignore') as file:
            solidity_code = file.read()
        contract_name, contract_content = extract_contract_with_events(solidity_code)
        # preprocess
        out = replace_symbols(contract_content)
        # feature extraction
        feat = Feature_extraction(out)
        colored(" Features Extracted >>>>>>>>", color='blue', on_color='on_grey')
        return feat

    def run_preprocessing(self):
        # Show loader (start progress bar)
        self.progress.start()
        # Run the actual preprocessing function (this could be slow, hence running in a thread)
        try:
            self.feat = self.Preprocess()
            self.feat = np.expand_dims(self.feat, axis=-1)
            self.feat = np.expand_dims(self.feat, axis=0)
            # When preprocessing is done, stop the progress bar and show result
            self.progress.stop()
            # Enable prediction button
            self.prediction_button.config(state="normal", bg="#80558c", fg="white", text="Predict")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred during preprocessing: {e}")
            self.progress.stop()

    def make_prediction(self):
        # Load the pre-trained model (assuming it's saved in 'proposed_model.h5')
        model = tf.keras.models.load_model('proposed_model.h5')
        out = np.argmax(model.predict(self.feat), axis=1)
        # Show prediction result
        if out == 0:
            self.prediction_button.config(text="No Vulnerability", fg="green")
        elif out == 1:
            self.prediction_button.config(text="Vulnerability Detected", fg="red")

    def refresh_interface(self):
        # Reset all UI components and clear file selection
        self.green_tick_var.set(False)
        self.green_tick_checkbox.config(state="disabled", fg="green", text="File Not Selected")
        self.file_path = ""
        self.progress.stop()
        self.prediction_button.config(state="disabled", text="Predict", fg="white")

    def exit_interface(self):
        # Close the interface
        self.root.quit()

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import threading
import numpy as np
import tensorflow as tf

class SolFileInterface:
    def __init__(self, root):
        self.root = root
        self.root.title("Solidity File Interface")

        # Set window size and background color
        self.root.geometry("500x500")
        self.root.config(bg='#88a2c4')

        # Define button width, height, and border thickness
        button_width = 20
        button_height = 2
        border_thickness = 3

        # Add file selection button
        self.select_file_button = tk.Button(self.root, text="Select .sol file", command=self.select_file,
                                            bg="white", fg="#80558c", font=("Arial", 12), width=button_width, height=button_height,
                                            borderwidth=border_thickness, relief="solid")
        self.select_file_button.pack(pady=20)

        # Green tick checkbox (Initially unchecked)
        self.green_tick_var = tk.BooleanVar(value=False)
        self.green_tick_checkbox = tk.Checkbutton(self.root, text="File Not Selected", variable=self.green_tick_var,
                                                  state="disabled", fg="white", bg="black", font=("Arial", 10))  # Set default to white
        self.green_tick_checkbox.pack(pady=10)

        # Progress bar for preprocessing (loader)
        self.progress = ttk.Progressbar(self.root, length=100, mode='indeterminate')
        self.progress.pack(pady=20)

        # Button for preprocessing
        self.preprocess_button = tk.Button(self.root, text="Preprocess", state="disabled", command=self.preprocess_file,
                                           bg="white", fg="#80558c", font=("Arial", 12), width=button_width, height=button_height,
                                           borderwidth=border_thickness, relief="solid")
        self.preprocess_button.pack(pady=20)

        # Prediction button (will show vulnerability or not)
        self.prediction_button = tk.Button(self.root, text="Predict", state="disabled", command=self.make_prediction,
                                           bg="white", fg="#80558c", font=("Arial", 12), width=button_width, height=button_height,
                                           borderwidth=border_thickness, relief="solid")
        self.prediction_button.pack(pady=10)

        # Refresh button
        self.refresh_button = tk.Button(self.root, text="Refresh", command=self.refresh_interface,
                                        bg="#c43138", fg="white", font=("Arial", 12), width=button_width, height=button_height,
                                        borderwidth=border_thickness, relief="solid")
        self.refresh_button.pack(pady=10)

        # Exit button
        self.exit_button = tk.Button(self.root, text="Exit", command=self.exit_interface, bg="#c43138", fg="white",
                                     font=("Arial", 12), width=button_width, height=button_height,
                                     borderwidth=border_thickness, relief="solid")
        self.exit_button.pack(pady=10)

        self.file_path = ""

    def select_file(self):
        # Open file dialog to select .sol file
        file_path = filedialog.askopenfilename(
            initialdir="Etherium Smart Contract/Etherium Smart Contract/smart contracts/block number dependency (BN)",
            filetypes=[("Solidity Files", "*.sol")])

        if file_path:
            self.file_path = file_path
            # Enable the green tick box and show that the file is selected
            self.green_tick_var.set(True)
            self.green_tick_checkbox.config(state="normal", fg="green", text="File Selected")  # Green after selection
            # Change the select file button color to green
            self.select_file_button.config(bg="green", fg="white")
            # Enable the preprocess button
            self.preprocess_button.config(state="normal", bg="#4CAF50", fg="white", text="Preprocess")

    def preprocess_file(self):
        threading.Thread(target=self.run_preprocessing).start()

    def Preprocess(self):
        with open(self.file_path, 'r', encoding='utf8', errors='ignore') as file:
            solidity_code = file.read()
        contract_name, contract_content = extract_contract_with_events(solidity_code)
        # preprocess
        out = replace_symbols(contract_content)
        # feature extraction
        feat = Feature_extraction(out)
        colored(" Features Extracted >>>>>>>>", color='blue', on_color='on_grey')
        return feat

    def run_preprocessing(self):
        # Show loader (start progress bar)
        self.progress.start()
        # Run the actual preprocessing function (this could be slow, hence running in a thread)
        try:
            self.feat = self.Preprocess()
            self.feat = np.expand_dims(self.feat, axis=-1)
            self.feat = np.expand_dims(self.feat, axis=0)
            # When preprocessing is done, stop the progress bar and show result
            self.progress.stop()
            # Change the preprocess button color to green and update text
            self.preprocess_button.config(bg="#80558c", fg="white", text="Preprocessed")
            # Enable prediction button
            self.prediction_button.config(state="normal", bg="#80558c", fg="white", text="Predict")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred during preprocessing: {e}")
            self.progress.stop()

    def make_prediction(self):
        # Load the pre-trained model (assuming it's saved in 'proposed_model.h5')
        model = tf.keras.models.load_model('proposed_model.h5')
        out = np.argmax(model.predict(self.feat), axis=1)
        # Show prediction result
        if out == 0:
            self.prediction_button.config(text="No Vulnerability", fg="green")
        elif out == 1:
            self.prediction_button.config(text="Vulnerability Detected", fg="red")

    def refresh_interface(self):
        # Reset all UI components and clear file selection
        self.green_tick_var.set(False)
        self.green_tick_checkbox.config(state="disabled", fg="white", text="File Not Selected")  # Reset to white
        self.file_path = ""
        self.progress.stop()
        self.prediction_button.config(state="disabled", text="Predict", fg="white")

        # Reset button colors to default after refresh
        self.select_file_button.config(bg="#80558c", fg="white")
        self.preprocess_button.config(bg="#80558c", fg="white", text="Preprocess")
        self.prediction_button.config(bg="#80558c", fg="white", text="Predict")

    def exit_interface(self):
        # Close the interface
        self.root.quit()


if __name__ == "__main__":
    root = tk.Tk()
    interface = SolFileInterface(root)
    root.mainloop()



if __name__ == "__main__":
    root = tk.Tk()
    interface = SolFileInterface(root)
    root.mainloop()

#
# if __name__ == "__main__":
#     root = tk.Tk()
#     interface = SolFileInterface(root)
#     root.mainloop()
