from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import pandas as pd
import json
from pydantic import BaseModel

app = FastAPI()

# CORS settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load tokenizer from a JSON file
with open('tokenizer.json', 'r') as f:
    data = json.load(f)
    tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(data)

# Define the data model for input using Pydantic
class TextData(BaseModel):
    title: str
    text: str

@app.post("/preprocess")
async def process_text(text_data: TextData):
    df = pd.DataFrame({'title': [text_data.title], 'text': [text_data.text]}, index=[0])
    df['combined_text'] = df['title'] + ' ' + df['text']

    # Tokenize and pad
    sequences = tokenizer.texts_to_sequences(df['combined_text'].tolist())
    padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=526)

    # Convert numpy array to list for JSON serialization
    return {"padded_sequence": padded_sequences.tolist()}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)
