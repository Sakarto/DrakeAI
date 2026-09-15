# DrakeAI (DrizzyAI)

Find a Drake song for any feeling.

You can try the live app using the website link on the right side of this GitHub page.

DrizzyAI is a mood-based Drake song recommender. Describe how you're feeling — a specific moment, a vague vibe, anything — and it matches you to the Drake song that fits best, using AI-generated song profiles and semantic similarity search.

## How it works

1. **Lyrics** are pulled from Genius for each song, album by album.
2. **Keyword profiles** are generated per song via Gemini with web search — researching critical reception, lyrical meaning, production style, and cultural context, then distilling that into a set of concrete keywords and phrases capturing mood, energy, imagery, and the kinds of moments each song fits.
3. **Embeddings** are generated for both the song's raw lyrics and its keyword profile using Gemini's embedding model, converting each into a vector.
4. When you describe your mood, that input is embedded the same way and compared against both the lyric and keyword embeddings for every song using cosine similarity. The two similarity scores are combined, and the song with the highest combined score is recommended.

Steps 1–3 are run offline ahead of time, one album at a time, to build the song dataset. The live app only handles step 4, so it stays fast and doesn't re-run the full pipeline on every use.

## Tech stack

- Python
- [Streamlit](https://streamlit.io/) — app interface
- [lyricsgenius](https://github.com/johnwmillr/LyricsGenius) — lyrics retrieval
- [Google Gemini API](https://ai.google.dev/) — research, keyword generation, and embeddings
- PyTorch — cosine similarity comparison

## Setup

1. Clone the repo
2. Install dependencies: `pip install -r requirements.txt`
3. Create a `.env` file with your API keys:

   GEMINI_API_KEY=your_key_here

   GENIUS_ACCESS_TOKEN=your_key_here

4. Run the app: `streamlit run DrizzyAI.py`

## Building the dataset

The dataset is built one album at a time by running `DrizzyAI_song_descriptions.py` with an `ALBUM_NAME` environment variable set to the album you want to process. Songs already present in `embedded_tracks.json` are automatically skipped, so the script can be re-run safely across multiple albums or interrupted runs.

## Status

The dataset is being rebuilt album by album under the new lyrics-and-keyword embedding approach. Standalone singles and loose tracks are not currently included.
