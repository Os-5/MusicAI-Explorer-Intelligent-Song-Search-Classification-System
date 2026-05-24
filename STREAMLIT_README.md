# Streamlit App (Spotify-style)

This adds a **Streamlit** UI on top of your existing project files:

- `songs_clustered.csv`
- `best_model.pkl`
- `embedding_model.pkl`

## Install (Windows PowerShell)

From the project root:

```powershell
py -3.13 -m pip install -r .\streamlit_requirements.txt
```

## Run

```powershell
cd "d:\my codes\Big data Project"
py -3.13 -m streamlit run .\streamlit_app.py
```

Then open the URL Streamlit prints (usually `http://localhost:8501`).

## Features

- **Category choose**: filter by `label`
- **Search engine**: typo-tolerant search by song / artist / text
- **Sorting system**: views / likes / A→Z
- **Add your song**: paste lyrics/text → predict category using your model
- **Modern UI**: Spotify-style dark theme + subtle animations

