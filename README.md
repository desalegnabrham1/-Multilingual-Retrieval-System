Multilingual Retrieval System
A simple Flask-based information retrieval web app for English and Amharic text.

It supports:

Local document search
BM25 and TF-IDF ranking
Optional query expansion
Fallback fuzzy search when exact matches are weak
Optional Google Custom Search integration
Basic evaluation with Precision@K and Recall@K

Project Structure
app.py: Main Flask app and routes
wsgi.py: WSGI entrypoint for production
api/index.py: Vercel Python entrypoint
ir_system/: Retrieval engine and text processing logic
data/: Corpus, queries, and relevance labels
templates/: HTML pages
static/: CSS styles

Requirements
Python 3.10+
pip
Dependencies are in requirements.txt:

Flask
gunicorn
Quick Start (Local)
Create and activate a virtual environment

python -m venv .venv
.venv\Scripts\activate

Install dependencies

pip install -r requirements.txt

Run the app

python app.py

Open in browser

http://127.0.0.1:5000

Optional Web Search Setup (Google CSE)
If you want web search results in addition to local results, set these environment variables:

GOOGLE_CUSTOM_SEARCH_KEY (or GOOGLE_API_KEY)
GOOGLE_CUSTOM_SEARCH_CX (or GOOGLE_CSE_ID)
PowerShell example:

Main Routes
GET / : Home page
POST /search : Search local corpus (and optional web search)
GET /dataset : Dataset overview
GET /evaluation?k=3 : Evaluation report


Deployment Notes
Procfile uses gunicorn with wsgi:app
vercel.json routes all traffic to index.py
Data Files
data/corpus.json: Searchable documents
data/test_queries.json: Evaluation/test queries
data/qrels.json: Relevance labels


License
Add your license here (for example, MIT).

