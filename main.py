import numpy as np
import streamlit as st
import pandas as pd
from crawl_data.crawl_cmt_from_ytb import Crawler
import asyncio
import json
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
def process(comments):
    with open('notebook/models/tokenizer.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    tokenizer = tokenizer_from_json(data)
    sequences = tokenizer.texts_to_sequences(comments)
    pad = pad_sequences(sequences, padding='post', maxlen=30, truncating='post')
    # print(tokenizer.word_index['<OOV>'])
    # exit(0)
    model = load_model('notebook/models/model.h5')
    predictions = model.predict(pad)
    return np.argmax(predictions, axis=1) + 1
data = pd.read_csv('data/cmt_ytb.csv')['text'].values
print(data)
result = process(data)
print(result)