# Fragrance Explorer — Claude Code Context

## What this is
AI-powered personal fragrance recommender. Matches perfumes by olfactory
note similarity using TF-IDF + cosine similarity, builds a personal profile
from user ratings, and uses Claude to explain recommendations in natural
language.

## Stack
Python · Streamlit · scikit-learn · Pandas · Anthropic API (claude-sonnet)

## Project structure
- app.py — Streamlit UI, tabs and layout
- recommender.py — TF-IDF vectorization, cosine similarity, profile building,
  exploration logic, quality scoring
- data_manager.py — dataset loading, ratings persistence
- config.py — accords, exploration styles, constants
- launch.bat — Windows launcher
- fragrantica_dataset.csv — dataset (not in repo, downloaded separately from Kaggle)

## Key logic
- Vectorization and similarity are in recommender.py — core of the project,
  do not refactor without explicit instruction
- Personal profile is built as a weighted average of rated fragrance vectors
- Three tabs implemented and working: Abbina note, Esplora stili, Qualità
- Scatter plot of personal collection (freshness vs depth, colored by rating)
  is implemented
- Enrich notes tab (LLM-powered web lookup for missing pyramids) is implemented

## LLM backend
Anthropic API (claude-sonnet). Used for natural language explanation of
recommendations and for the enrich notes feature.

## Rules
- Write all code and docstrings in English
- Never modify config files containing API keys or credentials
- Do not push sensitive data, keys, or personal preferences to git
- Dataset file is not tracked in git — do not add it
- Ask before changing the recommendation algorithm or scoring logic
- Follow existing module separation: UI in app.py, logic in recommender.py,
  data in data_manager.py