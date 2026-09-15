import lyricsgenius
from google import genai
from google.genai import types
from time import sleep
import json
from dotenv import load_dotenv
import os
#----------------------------------------------------------------------------------------------------------------------------------

# Setup:

load_dotenv()
genius_access_token = os.environ['GENIUS_ACCESS_TOKEN']
gemini_api_key = os.environ['GEMINI_API_KEY']

# Change this to the album you want to process this run:
album_name = os.environ['ALBUM_NAME']

embedded_tracks = {}
if os.path.exists("embedded_tracks.json"):
    with open("embedded_tracks.json", "r") as f:
        embedded_tracks = json.load(f)
else:
    with open("embedded_tracks.json", "w") as f:
        json.dump(embedded_tracks, f, ensure_ascii=False)

missed_tracks = []
if os.path.exists("missed_tracks.json"):
    with open("missed_tracks.json", 'r') as f:
        missed_tracks = json.load(f)
else:
    with open("missed_tracks.json", 'w') as f:
        json.dump(missed_tracks, f, ensure_ascii=False)
#----------------------------------------------------------------------------------------------------------------------------------

# Lyrics:

TOKEN = genius_access_token
genius = lyricsgenius.Genius(TOKEN, timeout=60, retries=3)

album = genius.search_album(album_name, "Drake", text_format="plain", fetch_lyrics=True)

lyrics = {}
for track_number, song in album.tracks:
    title = song.title
    if title in embedded_tracks.keys():
        continue
    if song.lyrics is None:
        missed_tracks.append(title)
        continue
    lyrics[title] = song.lyrics

with open("missed_tracks.json", 'w') as f:
    json.dump(missed_tracks, f, ensure_ascii=False)
#----------------------------------------------------------------------------------------------------------------------------------

# External Research Keywords:

client = genai.Client(api_key=gemini_api_key)

context_keywords = {}
keywords = 10
for song in lyrics.keys():
    context_keywords[song] = (client.models.generate_content(model="gemini-3.5-flash-lite", contents=f'You are building a keyword profile for the Drake song "{song}" for a semantic search system that matches songs to short, freeform descriptions of a person\'s mood or life moment.\n\nFirst, use web search to research this specific song — not generic facts about Drake or this album era, but details specific to "{song}" itself: its critical reception (e.g. Pitchfork, Rolling Stone, Complex, HotNewHipHop, or similar outlets), its lyrical meaning and any known backstory, its sonic/production style and tempo, and what it\'s specifically known for or associated with. Prioritize whatever makes this song distinct from other songs on the same album — tempo, samples, features, standout moments, specific reception. If you cannot find solid information on some of this, rely on what you do find rather than guessing or fabricating anything.\n\nThen, based only on what your research actually surfaced, generate exactly {keywords} keywords or short phrases (1-4 words each) that capture this song\'s specific identity. Cover a mix of: its emotional tone and mood, its energy or pace (e.g. restless, laid-back, triumphant, tense, upbeat, aggressive), concrete imagery or subject matter unique to it, and specific situations, settings, or life moments it genuinely fits — consider the full range of possibilities (morning, day, evening, night; solitary or social; active or restful; work, exercise, romance, celebration, conflict, etc.) and choose whichever domains actually fit this song\'s real content, not whichever is a safe default.\n\nBefore including any keyword referencing time of day (e.g. "late night," "midnight," "nocturnal," "morning") or a generic mood-atmosphere descriptor (e.g. "hazy," "atmospheric," "vibes"), require specific, direct evidence from the lyrics or research that this song is actually about or set in that time or mood — not just an overall tone that loosely fits. If you do not have that direct, specific evidence, choose a different, more concrete keyword instead. Do not use terms so generic they could apply to most Drake songs (e.g. "fame," "money," "relationships") unless paired with something specific that makes this song\'s take on that theme distinct. Avoid restating the song title or artist name.\n\nRespond with only the final list of exactly {keywords} keywords or phrases, one per line. Do not include your research notes, any explanation, commas, numbering, bullet points, or quotation marks — just the plain keywords or phrases, each on its own line.').text)
    sleep(10)
#----------------------------------------------------------------------------------------------------------------------------------

# Embedding Lyrics and Keywords:

lyric_embeddings = {}
for song, lyric in lyrics.items():
    lyric_embeddings[song] = (client.models._embed_content(model="gemini-embedding-001", contents=lyric).embeddings[0].values)
    sleep(1)
    
keyword_embeddings = {}
for title, keyword in context_keywords.items():
    keyword_embeddings[title] = (client.models._embed_content(model="gemini-embedding-001", contents=keyword).embeddings[0].values)
    sleep(1)

for title in lyrics.keys():
    embedded_tracks[title] = [lyric_embeddings[title], keyword_embeddings[title]]
    with open("embedded_tracks.json", "w") as f:
        json.dump(embedded_tracks, f, ensure_ascii=False)
#----------------------------------------------------------------------------------------------------------------------------------