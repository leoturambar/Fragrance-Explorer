# Fragrance Explorer — Claude Code Context

## What this is
AI-powered personal fragrance recommender. Matches perfumes by olfactory
note similarity using TF-IDF + cosine similarity, builds a personal profile
from user ratings, and uses Claude to explain recommendations in natural
language.

## Stack
Python · Streamlit · scikit-learn · Pandas · Anthropic API (claude-sonnet)

## Project structure
- app.py — Streamlit UI, all tab layout and interaction logic
- recommender.py — TF-IDF vectorization, cosine similarity, profile building,
  exploration logic, quality scoring
- matcher.py — dataset loading/merging, fuzzy candidate matching,
  note enrichment from dataset, confirmed_matches persistence
- ratings.py — HTML import, CSV persistence, add/edit/delete, migration logic
- enricher.py — DuckDuckGo search, Fragrantica scraping
- llm.py — LLM backend (Claude or Ollama), three prompt functions
- config.py — ACCORDS list, EXPLORATION_STYLES dict, rating constants
- launch.bat — Windows one-click launcher
- data/fragrances_clean.csv, data/fragrances_raw.csv — datasets (not in repo)

## Tab structure (4 tabs)
1. La mia lista — personal catalogue, filters, inline edit/delete
2. Profilo — stats, radar, scatter plot, LLM narrative, two coming-soon stubs
3. Scopri — radio toggle between Raccomandazioni and Esplora stili
4. Gestione — three inner st.tabs(): Aggiungi · Abbina note · Arricchisci note

## Ownership model (ratings.py / app.py)
Three independent fields per entry:
- status: mutually exclusive — Bottle | Decant | Sample | Tested | '' (none)
- bottle_candidate: bool — hidden/N/A in UI when status = Bottle
- watchlist: bool — freestanding bookmark, orthogonal to status

Migration: load_ratings() auto-converts old 'ownership' CSVs on first load
(Full Bottle→Bottle, Decant→Decant, Sample→Sample, Wishlist→watchlist=True).

## Key logic
- Vectorization and similarity are in recommender.py — do not refactor
  without explicit instruction
- Personal profile (build_personal_profile) uses a combined weight:
    normalized_rating × status_multiplier
  Status multipliers (_status_weight in recommender.py):
    Bottle                          → 1.5×
    Decant/Sample/Tested + BC=True  → 1.2×
    Decant/Sample/Tested            → 1.0×
    No status + watchlist only      → 0.0 (excluded from profile)
- Avoid penalty: fragrances rated ≤ 5.5 generate an avoid vector; cosine
  similarity to that vector is subtracted at 0.3× from every candidate score
- Exclusion in get_recommendations(): two-pass — first by stored dataset_idx,
  then by brand+name string match — so unmatched personal entries are also
  excluded when "Escludi profumi già noti" is checked
- Graph visibility filter stored in st.session_state['scatter_filter'];
  selectbox lives in La mia lista, Profilo reads from session_state

## Confirmed matches (matcher.py)
confirmed_matches.csv stores (rating_idx, dataset_idx, ds_brand, ds_name).
enrich_ratings() resolves by normalized (brand, name) key via _make_dataset_lookup(),
falling back to the integer index only if the string key isn't found.
This makes matches stable across dataset reloads/reindexing.
Legacy rows with only dataset_idx still work via the integer fallback path.

## Session state keys (app.py)
- scatter_filter     — graph visibility selection (shared La mia lista↔Profilo)
- last_recs          — cached recommendation DataFrame (Scopri tab)
- last_exp           — cached exploration DataFrame (Scopri tab)
- expl_rec_{i}_{…}  — per-recommendation LLM explanation text
- profile_analysis   — LLM narrative profile text
- pending_duplicate  — form data awaiting duplicate confirmation in Gestione
- editing_{idx}      — inline edit form open state (La mia lista)
- confirm_delete_{…} — delete confirmation state (La mia lista)

## Known fixes (do not re-investigate)
1. Recommendations including known fragrances — fixed with two-pass exclusion
   (dataset_idx + brand/name string match) in get_recommendations() and
   get_exploration_recommendations().
2. "Spiega perché" button doing nothing — fixed by storing df_recs in
   st.session_state['last_recs'] so the expanders re-render on every rerun.
3. Duplicate-entry dialog buttons not firing — fixed by moving the confirmation
   dialog outside the st.form block; form data is stashed in
   st.session_state['pending_duplicate'] before st.rerun().
4. KeyError on dataset index in enrich_ratings() — fixed by storing brand+name
   alongside the integer index in confirmed_matches.csv and resolving via
   normalized string lookup instead of df_dataset.loc[integer].

## Rules
- Write all code and docstrings in English
- Never modify config files containing API keys or credentials
- Do not push sensitive data, keys, or personal preferences to git
- Dataset files are not tracked in git — do not add them
- Ask before changing the recommendation algorithm or scoring logic
- Follow existing module separation: UI in app.py, logic in recommender.py,
  data I/O in ratings.py and matcher.py
