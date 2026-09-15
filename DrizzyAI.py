from google import genai
from google.genai import types
import torch
import torch.nn.functional as F
import json
from dotenv import load_dotenv 
import os
import streamlit as st

load_dotenv()
gemini_api_key = os.environ['GEMINI_API_KEY']

st.title('DrakeAI: Find a Drake song for any feeling!')
st.subheader('Take the advice and give it a listen :)')
#----------------------------------------------------------------------------------------------------------------------------------

# User Input/Embedding:

client = genai.Client(api_key=gemini_api_key)

user_response = st.text_input("What best describes how you are feeling right now?", "Type here...")
embedded_user_response = client.models._embed_content(model="gemini-embedding-001", contents=user_response)
#----------------------------------------------------------------------------------------------------------------------------------

# Compare Embedding and Track Selection:

with open("embedded_tracks.json", 'r') as f:
    embtracks = json.load(f)

user_tensor = torch.tensor(embedded_user_response.embeddings[0].values).unsqueeze(0)


lyric_similarities = []
keyword_similarities = []
for track, [lyric_embedding, keyword_embedding] in embtracks.items():
    lyric_tensor = torch.tensor(lyric_embedding).unsqueeze(0)
    keyword_tensor = torch.tensor(keyword_embedding).unsqueeze(0)
    lyric_similarities.append((F.cosine_similarity(user_tensor, lyric_tensor)).item())
    keyword_similarities.append((F.cosine_similarity(user_tensor, keyword_tensor)).item())

k = 0
similarities = []
for track in lyric_similarities:
    similarities.append(track + keyword_similarities[k])
    k += 1

highest_similarity_spot = similarities.index(max(similarities))

if st.button("Submit"):
    st.success("Try listening to " + list(embtracks.keys())[highest_similarity_spot] + ".")
#----------------------------------------------------------------------------------------------------------------------------------
