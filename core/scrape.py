import requests
from bs4 import BeautifulSoup
import time
from random import choice, uniform

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:90.0) Gecko/20100101 Firefox/90.0",
]

def scrape_content(url):
    """
    Scrape content from the given URL.
    """
    headers = {"User-Agent": choice(USER_AGENTS)}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        content = ' '.join(p.get_text() for p in soup.find_all('p'))
        time.sleep(uniform(2, 5))  # Delay to mimic human browsing
        return content.strip()
    except Exception as e:
        print(f"Error scraping URL {url}: {e}")
        return "Content could not be retrieved."
