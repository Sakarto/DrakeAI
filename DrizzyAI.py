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
embedded_file = os.environ['EMBEDDED_FILE']

st.title('DrizzyAI: Find a Drake song for any feeling!')
#----------------------------------------------------------------------------------------------------------------------------------

# User Input/Embedding:

client = genai.Client(api_key=gemini_api_key)

user_response = st.text_input("What best describes how you are feeling right now?", "Type here...")
embedded_user_response = client.models._embed_content(model="gemini-embedding-001", contents=user_response)
#----------------------------------------------------------------------------------------------------------------------------------

# Compare Embedding and Track Selection:

with open(embedded_file, 'r') as f:
    embtracks = json.load(f)

user_tensor = torch.tensor(embedded_user_response.embeddings[0].values).unsqueeze(0)


similarities = []
for track, embedding in embtracks.items():
    track_tensor = torch.tensor(embedding).unsqueeze(0)
    similarities.append((F.cosine_similarity(user_tensor, track_tensor)).item())

highest_similarity_spot = similarities.index(max(similarities))

if st.button("Submit"):
    st.success("Try listening to " + list(embtracks.keys())[highest_similarity_spot] + ".")
#----------------------------------------------------------------------------------------------------------------------------------
