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
        json.dump(embedded_tracks, f)

missed_tracks = []
if os.path.exists("missed_tracks.json"):
    with open("missed_tracks.json", 'r') as f:
        missed_tracks = json.load(f)
else:
    with open("missed_tracks.json", 'w') as f:
        json.dump(missed_tracks, f)
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
    json.dump(missed_tracks, f)
#----------------------------------------------------------------------------------------------------------------------------------

# External Research:

client = genai.Client(api_key=gemini_api_key)

external_research = []
for song, lyric in lyrics.items():
    external_research.append(client.models.generate_content(model="gemini-3.1-flash-lite", contents=f'You are researching the Drake song "{song}" (from the album "{album_name}") to understand its specific critical reception, meaning, and cultural context — not generic observations that could apply to any Drake song from this era.  Use web search to find information on: 1. Critical reception — how reviewers/critics specifically described THIS song\'s sound, mood, or quality (e.g., Pitchfork, Rolling Stone, Complex, HotNewHipHop, or similar outlets) 2. Lyrical meaning and interpretation — what THIS song is specifically about, any known backstory or context behind why it was written 3. Sonic/production description — how THIS song\'s beat, tempo, or production is specifically described, including anything that distinguishes it from other tracks on the same album 4. Cultural context or reputation — what THIS song specifically is known for, how fans commonly talk about it, or moments/scenarios it\'s specifically associated with  Prioritize any details that make this song distinct from other songs on the same album — tempo differences, unusual samples, guest features, standout lyrical moments, or specific reception it received. Summarize your findings in clear, factual notes organized under those four categories. If you can\'t find solid information for a category, say so rather than guessing. Do not fabricate sources or claims — only report what your search actually surfaces.').text)
    sleep(10)
#----------------------------------------------------------------------------------------------------------------------------------

# Writing Description:

song_descriptions = {}
i = 0
for song, lyric in lyrics.items():
    song_descriptions[song] = (client.models.generate_content(model="gemini-3.1-flash-lite", contents=f"You are creating a rich descriptive profile of the Drake song \"{song}\" for a music recommendation system. This profile will be compared against everyday life scenarios and moods to find the best song match — so it needs to be specific enough to this song that it couldn't be mistaken for a different song on the same album.  Here are the song's lyrics: {lyric}  Here is research on the song's reception, meaning, sound, and context: {external_research[i]}  Before writing, identify what makes this song specifically distinct — its actual tempo, its actual lyrical content, its actual mood — rather than defaulting to general assumptions about the artist or album era. If the song genuinely is a late-night, moody track, say so — accuracy matters more than forced variety. But do not default to \"late-night,\" \"3 AM,\" \"hazy,\" \"nocturnal,\" or \"atmospheric\" unless the lyrics or research actually support that specific characterization for this specific song.  Write a single dense paragraph (6-10 sentences) that captures: - The overall mood and emotional tone, grounded in what's actually in the lyrics/research (not a generic assumption) - The energy/pace (e.g., high-energy, laid-back, tense, restless, triumphant, upbeat, aggressive, playful) — pick whatever is actually true, don't default to the same descriptor every time - Key imagery or themes specific to this song's lyrics - The sonic/production feel, if mentioned in the research - Situations, times of day, or life moments this song could realistically fit — consider the full range of possibilities (morning, day, evening, night; solitary or social; active or restful; work, exercise, romance, celebration, etc.) and choose whichever actually fits this song's real content, rather than defaulting to nighttime/solitary scenarios by habit  Write it as a flowing descriptive paragraph, not a list or summary of the plot. Avoid simply restating the lyrics — translate them into mood and scenario language. Do not fabricate details that aren't supported by the lyrics or research notes provided.").text)
    i += 1
    sleep(10)
#----------------------------------------------------------------------------------------------------------------------------------

# Embedding Tracks:

for song, description in song_descriptions.items():
    embedded_tracks[song] = client.models._embed_content(model="gemini-embedding-001", contents=description).embeddings[0].values
    with open("embedded_tracks.json", "w") as f:
        json.dump(embedded_tracks, f)
#----------------------------------------------------------------------------------------------------------------------------------