# 🌸 Fragrance Explorer

> AI-powered personal fragrance explorer — recommendations, olfactory profile, and style exploration based on your own ratings.

---

## What it does

Fragrance Explorer is a local web app that helps you navigate the world of perfumery starting from what you already know you like. It matches fragrances by olfactory note similarity, builds a personal profile from your ratings, and uses an LLM to explain *why* a recommendation fits your taste.

### Features

| Tab | Description |
|-----|-------------|
| **Abbina note** | Match fragrances from the dataset to a set of olfactory notes. Get LLM-explained recommendations. |
| **Esplora stili** | Venture into new olfactory territories (Oud, Chypre, Fougère, Gourmand…) with adjustable exploration intensity and gender filter. |
| **Qualità** | Filter recommendations by quality score, penalizing fragrances far from your personal profile. |

---

## How it works

- The dataset (~20k fragrances from Fragrantica via Kaggle) is vectorized using **TF-IDF** on top, middle, and base notes.
- Similarity is computed via **cosine similarity**.
- A personal profile is built from your rated fragrances and used to bias recommendations.
- **Claude API** (claude-sonnet) generates natural-language explanations of each recommendation.

---

## Setup

### Requirements

- Python 3.10+
- Conda (recommended)
- Anthropic API key

### Installation

```bash
# 1. Clone the repo
git clone https://github.com/leoturambar/fragrance-explorer.git
cd fragrance-explorer

# 2. Create conda environment
conda create -n fragrance python=3.10
conda activate fragrance

# 3. Install dependencies
pip install streamlit pandas scikit-learn anthropic

# 4. Add your API key
# Create a .env file or set the environment variable:
export ANTHROPIC_API_KEY=your_key_here

# 5. Add the dataset
# Download from Kaggle (search: "fragrantica perfumes dataset")
# Place the CSV in the project root as: fragrantica_dataset.csv

# 6. Run
streamlit run app.py
```

Or just double-click `launch.bat` on Windows (see below).

---

## Launch from desktop (Windows)

Use the included `launch.bat` to start the app with a double-click — no terminal needed.

---

## Project structure

```
fragrance-explorer/
├── app.py              # Streamlit UI — tabs and layout
├── recommender.py      # Matching, exploration, quality scoring
├── data_manager.py     # Dataset loading, ratings persistence
├── config.py           # Accords, exploration styles, constants
├── launch.bat          # Windows launcher
├── README.md
└── LICENSE
```

---

## Dataset

This project uses a community-scraped Fragrantica dataset available on Kaggle. It is not included in this repo. Download it separately and place it in the project root.

---

## Roadmap

- [ ] Scatter plot of personal collection (freshness vs depth, colored by rating)
- [ ] "Enrich notes" tab — LLM-powered web lookup for fragrances with missing pyramids
- [ ] Delete entry from personal list

---

## License

MIT — see [LICENSE](LICENSE)
