# Wikipedia Speedrunner using Semantic Embeddings

A Python project motivated by Youtuber Green Code's video on a wikipedia speedrunner, uses semantic embeddings to go from an initial starting wikipedia article to a specified target article.
by selecting links based on semantic similarity using transformer embeddings.

---

## How It Works

1. Start from a Wikipedia page
2. Extract all valid outgoing links
3. Cache links found in the page for future use
4. Embed link titles using **FastEmbed**
5. Rank links by cosine similarity to the target article
6. Navigate to the most semantically relevant link
7. Repeat until the target page is reached

---

## Technologies Used

- **Python 3.10**
- **Wikipedia API**
- **FastEmbed (BAAI/bge-small-en-v1.5)**
- **NumPy**

---

## Future Plans

- Agent for speed ✔️
    - Caching ✔️
    - FastEmbeds ✔️
- Agent for fewest clicks
    - Graph Traversal
