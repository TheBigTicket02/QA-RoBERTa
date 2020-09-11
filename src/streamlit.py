import streamlit as st
import torch
from inference import create_loader, predict
from model import TweetModel

PATH = "../input/qamodel/model.pth"

@st.cache
def main(path):
    model = torch.load(path, map_location=torch.device('cpu'))
    model.eval()
    return model

st.title('Tweet Sentiment Extraction')

sentiment = st.radio("Pick sentiment", ('Positive', 'Neutral', 'Negative'))

tweet = st.text_area('Enter tweet')

extract = st.button('Extract support phrases')
if extract:
    model = main(PATH)
    loader = create_loader(sentiment, tweet)
    output = predict(loader, model)

phrases = st.text_area('Support phrases', value=output[0])