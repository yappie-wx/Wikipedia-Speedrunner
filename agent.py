import wikipediaapi
from fastembed import TextEmbedding
import numpy as np
import time
from typing import List
import json

class WikiClient:
    """Handles Wikipedia API interactions."""

    def __init__(self, lang: str = "en"):
        self.wiki = wikipediaapi.Wikipedia(
            language=lang,
            user_agent="WikiSpeedrunEmbedBot/1.0 (your-email@example.com)"
        )

    def page_exists(self, title: str) -> bool:
        return self.wiki.page(title).exists()

    def get_valid_links(self, title: str) -> List[str]:
        page = self.wiki.page(title)
        if not page.exists():
            return []

        return [
            link for link in page.links
            if ":" not in link and link != title
        ]

class Embedder:
    """Embeds text and ranks similarity to a target concept."""
    
    def __init__(self, target_word: str):
        self.model = TextEmbedding("BAAI/bge-small-en-v1.5")
        self.target_word = target_word
        self.target_vector = self._embed([target_word], is_query=True)[0]

    def _embed(self, texts: List[str], is_query: bool) -> np.ndarray:
        prefix = "query: " if is_query else "passage: "
        texts = [prefix + t for t in texts]
        return np.stack(list(self.model.embed(texts)))

    def get_most_similar(self, texts: List[str], top_k: int = 5):
        if not texts:
            return []

        candidate_vectors = self._embed(texts, is_query=False)
        similarities = candidate_vectors @ self.target_vector

        ranked = sorted(
            zip(texts, similarities),
            key=lambda x: x[1],
            reverse=True
        )

        return ranked[:top_k]
    
def cache_links(
        title: str, 
        links: List[str]
):
    with open("cache.json", 'r', encoding='utf-8') as f:
        loaded = json.load(f)

    if title not in loaded.keys():
        loaded[title] = links

        with open("cache.json", 'w', encoding='utf-8') as f:
            json.dump(loaded, f, ensure_ascii=False, indent=4)
    else:
        print("Already exists")

def run_wiki_speedrun(
    start: str,
    target: str,
    max_steps: int = 100
):
    # Initialize Speedrunner
    wiki = WikiClient()
    embedder = Embedder(target_word=target)

    visited = set()
    current = start
    steps = 0

    start_time = time.perf_counter() # Monitor time taken

    while current != target and steps < max_steps:
        print(f"Current page: {current}")
        visited.add(current)

        with open("cache.json", 'r', encoding='utf-8') as f:
            loaded = json.load(f)

        if current not in loaded.keys():
            links = wiki.get_valid_links(current)
            cache_links(current, links)
        else:
            links = loaded[current]
        
        ranked_links = embedder.get_most_similar(links)

        if not ranked_links:
            print("No valid links found. Stopping.")
            break

        for link, _ in ranked_links:
            if link not in visited:
                current = link
                steps += 1
                break
        else:
            print("All candidates already visited. Stopping.")
            break

    elapsed = time.perf_counter() - start_time

    if current == target:
        print(f"\nReached '{target}' in {steps} steps")
    else:
        print("\nTarget not reached")

    print(f"Time taken: {elapsed:.2f} seconds")

if __name__ == "__main__":
    run_wiki_speedrun(
        start="Five Nights at Freddy's",
        target="Volleyball"
    )
