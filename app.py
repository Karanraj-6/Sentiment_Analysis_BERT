import streamlit as st
import torch
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
import boto3
import os

# AWS S3 Configuration
s3 = boto3.client('s3')
bucket_name = 'sentimentanalysiskaran'  # Update this to your actual S3 bucket name

# List of model files to download from S3
files = [
    {"s3_key": "sentiment_model/config.json", "local_path": "sentiment_model/config.json"},
    {"s3_key": "sentiment_model/model.safetensors", "local_path": "sentiment_model/model.safetensors"},
    {"s3_key": "sentiment_model/special_tokens_map.json", "local_path": "sentiment_model/special_tokens_map.json"},
    {"s3_key": "sentiment_model/tokenizer_config.json", "local_path": "sentiment_model/tokenizer_config.json"},
    {"s3_key": "sentiment_model/vocab.txt", "local_path": "sentiment_model/vocab.txt"}
]

# Create the directory for model files if it doesn't exist
if not os.path.exists('sentiment_model'):
    os.makedirs('sentiment_model')

def download_from_s3():
    """Download model files from S3."""
    for file in files:
        try:
            print(f"Downloading {file['s3_key']} from S3...")
            s3.download_file(bucket_name, file['s3_key'], file['local_path'])
            print(f"{file['s3_key']} downloaded successfully.")
        except Exception as e:
            print(f"Error downloading {file['s3_key']}: {str(e)}")

# Download model files from S3
download_from_s3()

# Load the model and tokenizer
model = DistilBertForSequenceClassification.from_pretrained('sentiment_model')
tokenizer = DistilBertTokenizer.from_pretrained('sentiment_model')

def preprocess_input(input_text):
    """Preprocess the input text for the model."""
    encoded_input = tokenizer(input_text, padding='max_length', truncation=True, return_tensors='pt', max_length=512)
    return encoded_input

# Custom CSS for the colorful border for the text area
st.markdown(
    """
    <style>
    .custom-text-area {
        border: 3px solid;
        border-image: linear-gradient(to right, red, yellow, green, blue) 1;
        border-radius: 5px;
        padding: 10px;
        height: 200px;  /* Adjust height as needed */
        resize: none;   /* Disable resizing */
        width: 100%;    /* Full width */
        box-sizing: border-box; /* Ensure padding does not affect overall width */
    }
    </style>
    """, unsafe_allow_html=True
)

# Title of the Streamlit app
st.markdown("<h1 style='color: black;'>Sentiment Analysis with BERT</h1>", unsafe_allow_html=True)

# User input text area
user_input = st.text_area(
    "Enter your text:", 
    height=200, 
    key="input_text", 
    help="Type your text here...", 
    placeholder="Type your text here..."
)

# Apply the custom CSS directly to the text area
st.markdown(
    '<style>div[data-baseweb="textarea"] { border: 3px solid; border-image: linear-gradient(to right, red, yellow, green, blue) 1; border-radius: 5px; padding: 10px;}</style>', 
    unsafe_allow_html=True
)

# Button to trigger sentiment analysis
if st.button("Analyze"):
    if user_input:  # Check if user input is not empty
        preprocessed_input = preprocess_input(user_input)
        
        # Disable gradient calculation for inference
        with torch.no_grad():
            outputs = model(**preprocessed_input)
            predictions = torch.argmax(outputs.logits, dim=-1)
        
        # Determine sentiment: Positive or Negative
        sentiment = "Positive" if predictions.item() == 1 else "Negative"
        
        # Display the prediction with custom color styling
        if sentiment == "Positive":
            st.markdown(f"<h1 style='color: green;'>{sentiment}</h1>", unsafe_allow_html=True)
        else:
            st.markdown(f"<h1 style='color: red;'>{sentiment}</h1>", unsafe_allow_html=True)
    else:
        st.write("Please enter some text.")
