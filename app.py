import streamlit as st
import torch
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
import boto3
import os

s3 = boto3.client('s3')

bucket_name = 'sentimentanalysiskaran'
files = [
    {"s3_key": "sentiment_model/config.json", "local_path": "sentiment_model/config.json"},
    {"s3_key": "sentiment_model/model.safetensors", "local_path": "sentiment_model/model.safetensors"},
    {"s3_key": "sentiment_model/special_tokens_map.json", "local_path": "sentiment_model/special_tokens_map.json"},
    {"s3_key": "sentiment_model/tokenizer_config.json", "local_path": "sentiment_model/tokenizer_config.json"},
    {"s3_key": "sentiment_model/vocab.txt", "local_path": "sentiment_model/vocab.txt"}
]

if not os.path.exists('sentiment_model'):
    os.makedirs('sentiment_model')

def download_from_s3():
    for file in files:
        try:
            print(f"Downloading {file['s3_key']} from S3...")
            s3.download_file(bucket_name, file['s3_key'], file['local_path'])
            print(f"{file['s3_key']} downloaded successfully.")
        except Exception as e:
            print(f"Error downloading {file['s3_key']}: {str(e)}")

download_from_s3()

model = DistilBertForSequenceClassification.from_pretrained('sentiment_model')
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

def preprocess_input(input_text):
    encoded_input = tokenizer(input_text, padding='max_length', truncation=True, return_tensors='pt', max_length=512)
    return encoded_input

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
    }
    </style>
    """, unsafe_allow_html=True
)

st.markdown("<h1 style='color: black;'>Sentiment Analysis with BERT</h1>", unsafe_allow_html=True)

user_input = st.text_area(
    "Enter your text:", 
    height=200, 
    key="input_text", 
    help="Type your text here...",
    value="", 
    placeholder="Type your text here..."
)

st.markdown(
    '<style>div[data-baseweb="textarea"] { border: 3px solid; border-image: linear-gradient(to right, red, yellow, green, blue) 1; border-radius: 5px; padding: 10px;}</style>', 
    unsafe_allow_html=True
)

if st.button("Analyze"):
    if user_input:
        preprocessed_input = preprocess_input(user_input)
        
        with torch.no_grad():
            outputs = model(**preprocessed_input)
            predictions = torch.argmax(outputs.logits, dim=-1)
        
        sentiment = "Positive" if predictions.item() == 1 else "Negative"
        
        if sentiment == "Positive":
            st.markdown(f"<h1 style='color: green;'>{sentiment}</h1>", unsafe_allow_html=True)
        else:
            st.markdown(f"<h1 style='color: red;'>{sentiment}</h1>", unsafe_allow_html=True)
    else:
        st.write("Please enter some text.")
