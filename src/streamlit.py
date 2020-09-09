import streamlit as st

st.title('Tweet Sentiment Extraction')

sentiment = st.radio("Pick sentiment", ('Positive', 'Neutral', 'Negative'))

tweet = st.text_area('Enter tweet')

extract = st.button('Extract support phrases')

st.text_area('Support phrases')