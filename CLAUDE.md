# Fragrance Explorer — Claude Code Context

## What this is
AI-powered personal fragrance recommender. Matches perfumes by olfactory
note similarity using TF-IDF + cosine similarity, builds a personal profile
from user ratings, and uses Claude to explain recommendations in natural
language.

## Stack
Python · Streamlit · scikit-learn · Pandas · Anthropic API or local Ollama (see llm.py)

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
2. Profilo — stats, liked-vs-disliked radar, scatter (own filter), gap analysis,
   time drift (needs date_added), LLM narrative. Uses tried() entries only.
3. Scopri — segmented control: Dal mio profilo · Simile a… · Esplora uno stile
4. Gestione — inner st.tabs(): Aggiungi · Abbina note · Arricchisci note · Salute dati

## Ownership model (ratings.py / app.py)
Three independent fields per entry:
- status: mutually exclusive — Bottle | Decant | Sample | Tested | '' (none)
- bottle_candidate: bool — hidden/N/A in UI when status = Bottle
- watchlist: bool — freestanding bookmark, orthogonal to status

Watchlist-only entries (status '' + watchlist) may have rating = NaN ("not
tried yet"). ratings.tried(df) filters them out — use it for stats, charts,
profile and LLM context.

## Data schema (data/my_ratings.csv — single source of truth)
- id: stable 8-char hex, generated on load if missing. ALWAYS address entries
  by id (never by row position — positions shift on delete).
- date_added: ISO date for entries added since Nov 2026, '' for legacy.
- Notes: top_notes, middle_notes, base_notes, main_accords, notes_source
  ('dataset' | 'web_enriched' | 'manual'), notes_url.
- Dataset match: match_status ('matched' | 'skipped' | ''), ds_brand, ds_name.
- Derived on load, never saved: matched (= has any notes),
  dataset_idx (resolved from ds_brand/ds_name by matcher.resolve_dataset_indices).

Migration: load_ratings() auto-converts old 'ownership' CSVs on first load
(Full Bottle→Bottle, Decant→Decant, Sample→Sample, Wishlist→watchlist=True),
adds missing columns/ids, and fixes legacy "(Decant)" values stored in 'form'.

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
- Exclusion: matcher.known_dataset_indices() = resolved dataset_idx + fuzzy
  brand (prefix/word) and cleaned-name match (token_sort ≥ 95, brand,
  concentration, edition and year words stripped). Flankers stay recommendable.
  app.py caches it (get_known_idx) and passes known_idx to the recommenders.
- Gender filter uses dataset column gender_norm ('male'|'female'|'unisex',
  from "for men"/"for women"/"for women and men"); GENDER_OPTIONS maps to tuples.
- prepare_dataset() precomputes quality_score + note_string; app caches it with
  st.cache_resource. Recommenders accept raw or prepared frames.
- get_similar_to(rid, …) forces the reference rating to 10 so it never
  becomes its own avoid vector.

## Dataset matches (matcher.py / ratings.py)
Matches live on each entry (ds_brand/ds_name). confirmed_matches.csv is gone
(old positional file kept as data/confirmed_matches.legacy.csv).
- ratings.apply_match(): takes dataset notes only if the row has a pyramid,
  otherwise keeps existing (e.g. Fragrantica) notes and stores the reference.
- Matching one entry never rewrites other entries.
- matcher.match_health() powers Gestione → Salute dati.
- Fragrantica returns 403 to automated requests; manual note entry exists.

## Session state keys (app.py)
- scatter_filter     — scatter visibility selection (Profilo)
- last_{mode}        — {'df', 'style'} result per Scopri mode (profilo|simile|stile)
- expl_rec_{mode}_…  — per-recommendation LLM explanation text
- exp_explanation    — on-demand LLM explanation for Esplora uno stile
- profile_analysis   — LLM narrative profile text
- pending_duplicate  — add_rating kwargs awaiting duplicate confirmation
- just_added         — id of the entry just added (quick-match box)
- editing_{id} / confirm_delete_{id} — card edit/delete state (La mia lista)
- table_editor       — st.data_editor state (La mia lista, Tabella view)

Gotcha: selectbox format_func lambdas must not close over variables that are
reassigned later in the script (e.g. `pool`) — bind a dict of labels instead.

## Known fixes (do not re-investigate)
1. Recommendations including known fragrances — fixed with two-pass exclusion
   (dataset_idx + brand/name string match) in get_recommendations() and
   get_exploration_recommendations().
2. "Spiega perché" button doing nothing — fixed by storing df_recs in
   st.session_state['last_recs'] so the expanders re-render on every rerun.
3. Duplicate-entry dialog buttons not firing — fixed by moving the confirmation
   dialog outside the st.form block; form data is stashed in
   st.session_state['pending_duplicate'] before st.rerun().
4. Notes from the wrong fragrance (positional confirmed_matches + reindexed
   dataset; enrich_ratings() rewrote every entry on each match) — fixed Nov
   2026 by storing matches per entry by id. Data was repaired; backups in
   data/*.backup-2026-11-06.csv.
5. enricher print() with "→" crashed on the Windows cp1252 console.

## Rules
- Write all code and docstrings in English
- Never modify config files containing API keys or credentials
- Do not push sensitive data, keys, or personal preferences to git
- Dataset files are not tracked in git — do not add them
- Ask before changing the recommendation algorithm or scoring logic
- Follow existing module separation: UI in app.py, logic in recommender.py,
  data I/O in ratings.py and matcher.py
