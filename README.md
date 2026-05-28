🎵 Intelligent Music Discovery & Recommendation System
An end-to-end Machine Learning pipeline and interactive web application that classifies, clusters, and matches songs using natural language processing (NLP). The platform allows users to find music using traditional text-matching metadata patterns or semantic lyric analysis powered by Sentence Transformers.

🚀 Features
Dual-Engine Search Mode:

Fuzzy Matching / Text Search: Powered by rapidfuzz token-ratio matching across song names, artists, categories, and clusters.

NLP Semantic Search: Encodes queries on-the-fly and runs cosine similarity math against a pre-computed sentence embedding matrix.

Song Classification UI: Submit custom lyrics, titles, or general descriptions via an intuitive input form to dynamically embed textual data and trigger real-time category classification.

High-Performance Caching: Leverages Streamlit’s @st.cache_resource and @st.cache_data structures to maintain immediate execution across vast dataset lookups.

Spotify-Inspired Aesthetics: Custom-injected CSS styling layered with glowing background radial gradients, responsive result rows, custom inline metrics, and direct YouTube reference links.

📂 Project Architecture
Code snippet
├── music recommendation system2.ipynb   # Data ingestion, preprocessing, & training pipeline
├── streamlit_app.py                      # Interactive UI and deployment-ready search app
├── best_model.pkl                       # Serialized optimal classification model
├── embedding_model.pkl                  # Serialized SentenceTransformer encoder model
└── songs_clustered.csv                  # Main dataset containing song profiles and metrics

⚙️ Data Pipeline & Model Training
The foundational processing outlined in the Jupyter Notebook splits structural preparation into explicit stages:

Preprocessing & Tokenization: Ingests your primary track collection (spotify_millsongdata.csv), cleans lyric bodies using regex formatting, normalizes spacing, strips boilerplate English NLTK stopwords, and normalizes lemmas via a WordNetLemmatizer loop.

Feature Engineering / Embeddings: Implements sentence vector extraction utilizing the SentenceTransformer framework (all-MiniLM-L6-v2) to turn lyrics into numerical representations.

Model Selection: Benchmarks diverse classification classifiers, optimizes baseline validation performance, and exports the final model stack via joblib for application inference.

🛠️ Installation & Setup
Prerequisites
Ensure your localized system runs Python 3.9+ environments.

1. Clone the Repository
Bash
git clone https://github.com/your-username/music-recommendation-system.git
cd music-recommendation-system
2. Install Dependencies
Install the required packages using pip:

Bash
pip install streamlit pandas numpy joblib scikit-learn sentence-transformers rapidfuzz nltk
3. Initialize NLP Corpora
Download the necessary NLTK components used in text normalization:

Bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordne

🖥️ Running the Application
Place your generated data and serialized files (songs_clustered.csv, best_model.pkl, and embedding_model.pkl) into the root path directory alongside the app script. Start the application by executing:

Bash
streamlit run streamlit_app.py
