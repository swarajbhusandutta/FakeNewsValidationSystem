import time
import requests
from bs4 import BeautifulSoup
import random
import spacy

# Load spaCy model for NLP processing
nlp = spacy.load("en_core_web_sm")

# Define user-agents to avoid bot detection
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
]

# List of search engines for rotation
SEARCH_ENGINES = {
    "google": "https://www.google.com/search?q=",
    "bing": "https://www.bing.com/search?q=",
    "duckduckgo": "https://duckduckgo.com/?q="
}


def extract_key_phrases(text):
    """Extracts important key phrases from the input text using NLP."""
    doc = nlp(text)
    phrases = [chunk.text for chunk in doc.noun_chunks]
    return phrases if phrases else [text]  # Fallback to original text


def construct_search_query(text):
    """Constructs a search query ensuring words appear together."""
    key_phrases = extract_key_phrases(text)
    return ' "'.join([phrase for phrase in key_phrases]) + '"'


def search_news(query):
    """Performs a search query across multiple search engines and scrapes results."""
    results = {}
    for engine, base_url in SEARCH_ENGINES.items():
        try:
            url = base_url + requests.utils.quote(query)
            headers = {"User-Agent": random.choice(USER_AGENTS)}
            response = requests.get(url, headers=headers, timeout=5)
            time.sleep(random.uniform(3, 6))  # Sleep to avoid bot detection

            if response.status_code == 200:
                soup = BeautifulSoup(response.text, "html.parser")
                links = [a['href'] for a in soup.find_all("a", href=True) if "http" in a['href']]
                results[engine] = links[:10]  # Store top 10 results per engine
        except Exception as e:
            print(f"Error searching {engine}: {e}")
    return results


def perform_search(content):
    """Main function to execute the search algorithm."""
    query = construct_search_query(content)
    search_results = search_news(query)
    return search_results
