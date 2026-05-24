MusicAI Explorer

MusicAI Explorer is a Streamlit-based music search and classification app that uses NLP embeddings, fuzzy matching, and machine learning to explore a music dataset in an interactive way.

Features
Hybrid search using NLP (semantic) and fuzzy matching
Search songs by title, artist, label, or cluster (supports typos)
Machine learning model for music classification
Lyrics viewer for each song
Fast performance using caching
Modern Streamlit UI
Tech Stack
Python
Streamlit
Pandas
NumPy
Scikit-learn
SentenceTransformers
RapidFuzz
Joblib
Project Structure

MusicAI/

app.py
songs_clustered.csv
best_model.pkl
embedding_model.pkl
requirements.txt
Installation
Clone the repository:
git clone https://github.com/USERNAME/REPO_NAME.git
cd REPO_NAME
Install dependencies:
pip install -r requirements.txt
Run the App

streamlit run app.py

Then open:
http://localhost:8501

How It Works
User enters a search query
System uses NLP embeddings + fuzzy matching to find best results
Results are ranked using a custom scoring system
ML model predicts music category from lyrics or text input
UI displays songs with metadata and lyrics view
Example Use Cases
Search songs with typos (e.g. “shackira”)
Find songs by mood or description
Explore dataset by labels or clusters
Classify lyrics into categories
