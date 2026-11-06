import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from matcher import load_dataset, get_candidates


# Ratings at or below this only feed the avoid vector (never the positive profile)
AVOID_MAX_RATING = 5.5

# How strongly "Favorisci la nicchia" down-weights popular fragrances (0-1)
NICHE_STRENGTH = 0.5
# Vote count at which a niche fragrance's rating is considered fully reliable
NICHE_FULL_CONFIDENCE_VOTES = 100


def build_note_string(row) -> str:
    """
    Combina top, middle, base notes e main accords in una stringa unica.
    Usata per la vettorizzazione TF-IDF.
    """
    parts = []
    for col in ['top_notes', 'middle_notes', 'base_notes', 'main_accords']:
        val = row.get(col, '')
        if val and str(val) != 'nan':
            parts.append(str(val))
    return ' '.join(parts).lower()


def _status_weight(row) -> float:
    """
    Returns multiplier based on ownership status and bottle_candidate flag.
    Bottle → 1.5x
    Decant/Sample/Tested + bottle_candidate → 1.2x
    Decant/Sample/Tested → 1.0x
    No status + watchlist only → 0.0 (excluded from profile)
    """
    status = str(row.get('status', '')).strip()
    bc = bool(row.get('bottle_candidate', False))
    watchlist = bool(row.get('watchlist', False))
    if status == 'Bottle':
        return 1.5
    if status in ('Decant', 'Sample', 'Tested'):
        return 1.2 if bc else 1.0
    if not status and watchlist:
        return 0.0
    return 1.0


def build_personal_profile(df_ratings: pd.DataFrame) -> dict:
    """
    Calcola il profilo olfattivo personale come media pesata per rating e status.
    Restituisce dict con:
        - note_string: stringa pesata di note preferite
        - top_notes: note più amate (rating >= 8)
        - avoid_notes: note associate a rating bassi (<= AVOID_MAX_RATING)

    Fragrances rated <= AVOID_MAX_RATING feed ONLY the avoid vector: they no
    longer add a positive contribution to note_string (previously every
    entry, even a 3.5, was repeated at least once in the positive profile).

    Pesi status:
        Bottle → 1.5x | Decant/Sample/Tested + BC → 1.2x
        Decant/Sample/Tested → 1.0x | Watchlist senza status → escluso
    """
    df = df_ratings[df_ratings['matched'] == True].copy()
    if df.empty:
        return {'note_string': '', 'top_notes': '', 'avoid_notes': ''}

    # esclude voci watchlist-only (nessuno status)
    status_col   = df.get('status',    pd.Series('',    index=df.index)).fillna('').astype(str).str.strip()
    watchlist_col = df.get('watchlist', pd.Series(False, index=df.index)).fillna(False)
    df = df[~((status_col == '') & watchlist_col)]

    if df.empty:
        return {'note_string': '', 'top_notes': '', 'avoid_notes': ''}

    # positive profile: only fragrances above the avoid threshold.
    # Rating normalized to 0-1 between the threshold and the best rating,
    # then multiplied by the status multiplier.
    df_pos = df[df['rating'] > AVOID_MAX_RATING].copy()
    r_max = df_pos['rating'].max() if not df_pos.empty else AVOID_MAX_RATING + 1
    df_pos['_rating_w']    = (df_pos['rating'] - AVOID_MAX_RATING) / (r_max - AVOID_MAX_RATING)
    df_pos['_status_mult'] = df_pos.apply(_status_weight, axis=1) if not df_pos.empty else 1.0
    df_pos['weight']       = df_pos['_rating_w'] * df_pos['_status_mult']

    weighted_parts = []
    for _, row in df_pos.iterrows():
        note_str = build_note_string(row)
        if not note_str.strip():
            continue
        repeats = max(1, round(row['weight'] * 3))
        weighted_parts.extend([note_str] * repeats)

    df_top = df[df['rating'] >= 8.0]
    top_notes = ' '.join(build_note_string(r) for _, r in df_top.iterrows())

    df_avoid = df[df['rating'] <= AVOID_MAX_RATING]
    avoid_notes = ' '.join(build_note_string(r) for _, r in df_avoid.iterrows())

    return {
        'note_string': ' '.join(weighted_parts),
        'top_notes':   top_notes,
        'avoid_notes': avoid_notes,
    }


def build_brand_scores(df_dataset: pd.DataFrame) -> dict:
    import numpy as np
    df = df_dataset.copy()
    df['Rating Value'] = pd.to_numeric(df['Rating Value'], errors='coerce')
    df['Rating Count'] = pd.to_numeric(df['Rating Count'], errors='coerce')
    df = df.dropna(subset=['Rating Value', 'Rating Count'])
    df['brand_lower'] = df['Brand'].str.lower().str.strip()

    brand_scores = {}
    for brand, group in df.groupby('brand_lower'):
        avg_rating = group['Rating Value'].mean()
        # usa mediana recensioni per profumo — meno sensibile ai bestseller virali
        median_reviews = group['Rating Count'].median()
        # rating medio è il fattore principale, recensioni solo tiebreaker
        brand_scores[brand] = avg_rating * np.log1p(median_reviews)

    if brand_scores:
        max_score = max(brand_scores.values())
        brand_scores = {k: v / max_score for k, v in brand_scores.items()}

    return brand_scores


def compute_quality_score(row, brand_scores: dict) -> float:
    """
    Score qualità combinato per un singolo profumo.
    B: rating ponderato per log(recensioni)
    C: boost brand reputation
    """
    import numpy as np
    rating = pd.to_numeric(row.get('Rating Value', 0), errors='coerce') or 0
    count = pd.to_numeric(
        str(row.get('Rating Count', 0)).replace(',', ''), errors='coerce'
    ) or 0
    brand  = str(row.get('Brand', '')).lower().strip()

    score_b = rating * np.log1p(count)   # qualità individuale pesata per popolarità
    score_b = score_b / 46.0   # normalizza approssimativamente — max teorico è ~5 * log(10000+1) ≈ 46
    score_c = brand_scores.get(brand, 0)   # reputazione brand

    # combina: 70% score individuale, 30% brand reputation
    return 0.7 * score_b + 0.3 * score_c


def prepare_dataset(df_dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Adds 'quality_score' and 'note_string' to the dataset. Expensive (~70k rows),
    so callers should cache the result; already-prepared frames pass through.
    """
    if {'quality_score', 'note_string', 'popularity'} <= set(df_dataset.columns):
        return df_dataset.copy()
    df_ds = df_dataset.copy()

    # filtro qualità
    brand_scores = build_brand_scores(df_dataset)
    df_ds['quality_score'] = df_ds.apply(
        lambda r: compute_quality_score(r, brand_scores), axis=1
    )
    max_q = df_ds['quality_score'].max() or 1
    df_ds['quality_score'] = df_ds['quality_score'] / max_q

    # popularity (0-1, log of vote count) and a rating-only quality for niche mode
    votes = pd.to_numeric(df_ds['Rating Count'].astype(str).str.replace(',', ''),
                          errors='coerce').fillna(0)
    rating = pd.to_numeric(df_ds['Rating Value'], errors='coerce').fillna(0)
    log_votes = np.log1p(votes)
    df_ds['popularity'] = log_votes / (log_votes.max() or 1)
    confidence = (log_votes / np.log1p(NICHE_FULL_CONFIDENCE_VOTES)).clip(upper=1)
    # rating stretched over the useful 3-5 range, so the quality slider bites
    own = ((rating - 3) / 2).clip(0, 1) * confidence
    # brand reputation = average rating of its fragrances (no vote-count bias),
    # only for brands with at least 3 rated fragrances
    brand = df_ds['Brand'].str.lower().str.strip()
    rated = rating.where(votes > 0)
    b_mean = rated.groupby(brand).transform('mean')
    b_n = rated.groupby(brand).transform('count')
    brand_q = ((b_mean - 3.5) / 1.0).clip(0, 1).where(b_n >= 3, 0.3).fillna(0.3)
    df_ds['niche_quality'] = 0.65 * own + 0.35 * brand_q

    # costruisce stringa note
    df_ds['note_string'] = df_ds.apply(
        lambda r: ' '.join(filter(None, [
            str(r.get('Top', '')) if pd.notna(r.get('Top')) else '',
            str(r.get('Middle', '')) if pd.notna(r.get('Middle')) else '',
            str(r.get('Base', '')) if pd.notna(r.get('Base')) else '',
            ' '.join(str(r.get(f'mainaccord{i}', ''))
                     for i in range(1, 6)
                     if pd.notna(r.get(f'mainaccord{i}')))
        ])).lower(),
        axis=1
    )
    return df_ds


def _filter_gender(df_ds: pd.DataFrame, gender_filter) -> pd.DataFrame:
    """gender_filter: 'all' / None, or one/several of 'male' | 'female' | 'unisex'."""
    if not gender_filter or gender_filter == 'all':
        return df_ds
    allowed = {gender_filter} if isinstance(gender_filter, str) else set(gender_filter)
    if 'gender_norm' in df_ds.columns:
        return df_ds[df_ds['gender_norm'].isin(allowed)]
    from matcher import normalize_gender
    return df_ds[df_ds['Gender'].map(normalize_gender).isin(allowed)]


def _exclude_known(df_ds: pd.DataFrame, df_ratings: pd.DataFrame,
                   df_dataset: pd.DataFrame, known_idx) -> pd.DataFrame:
    """
    Removes fragrances already in the personal list: confirmed dataset matches
    plus fuzzy brand+name matches (see matcher.known_dataset_indices).
    """
    if known_idx is None:
        from matcher import known_dataset_indices
        known_idx = known_dataset_indices(df_ratings, df_dataset)
    return df_ds[~df_ds.index.isin(known_idx)]


def _apply_quality(df_ds: pd.DataFrame, quality_pct: int, niche: bool = False) -> pd.DataFrame:
    """
    Quality filter + multiplier on 'similarity'.

    Default: quality_score = rating × log(votes) blended with brand reputation,
    which favours very popular fragrances.
    niche=True: quality is the rating itself, discounted only when there are
    too few votes to trust it (< NICHE_FULL_CONFIDENCE_VOTES); then similarity
    is reduced by up to NICHE_STRENGTH for the most-voted fragrances.
    """
    quality_weight = 1 - (quality_pct / 100)
    q = df_ds['niche_quality'] if niche else df_ds['quality_score']
    min_threshold = quality_weight * 0.5
    df_ds = df_ds[q >= min_threshold].copy()
    q = q.loc[df_ds.index]

    penalty = q ** (1 + quality_weight * 2)
    df_ds['similarity'] = df_ds['similarity'] * (
        (1 - quality_weight) + quality_weight * penalty
    )
    if niche:
        df_ds['similarity'] = df_ds['similarity'] * (1 - NICHE_STRENGTH * df_ds['popularity'])
    return df_ds


def _top_n(df_ds: pd.DataFrame, n: int) -> pd.DataFrame:
    """De-duplicates *before* cutting to n, so n results are always returned."""
    df_ds = df_ds.sort_values('similarity', ascending=False) \
                 .drop_duplicates(subset=['Brand', 'Perfume'])
    df_result = df_ds.head(n)[
        ['Brand', 'Perfume', 'Gender', 'Top', 'Middle', 'Base',
         'mainaccord1', 'mainaccord2', 'similarity']
    ].copy()
    df_result['similarity'] = df_result['similarity'].round(3)
    df_result.columns = ['brand', 'name', 'gender', 'top', 'middle',
                         'base', 'accord1', 'accord2', 'similarity']
    return df_result.reset_index(drop=True)


def get_recommendations(df_ratings: pd.DataFrame,
                        df_dataset: pd.DataFrame,
                        n: int = 10,
                        exclude_known: bool = True,
                        gender_filter='all',
                        quality_pct: int = 40,
                        known_idx: set | None = None,
                        niche: bool = False) -> pd.DataFrame:
    """
    Genera raccomandazioni dal dataset Kaggle basate sul profilo personale.
    df_dataset may be raw or already passed through prepare_dataset().
    known_idx: dataset indices to exclude (computed if None).
    """
    profile = build_personal_profile(df_ratings)

    if not profile['note_string']:
        return pd.DataFrame()

    df_ds = prepare_dataset(df_dataset)

    # filtri
    df_ds = _filter_gender(df_ds, gender_filter)
    if exclude_known:
        df_ds = _exclude_known(df_ds, df_ratings, df_dataset, known_idx)

    df_ds = df_ds[df_ds['note_string'].str.strip() != '']

    if df_ds.empty:
        return pd.DataFrame()

    # TF-IDF + cosine similarity — calcolato dopo tutti i filtri
    corpus = [profile['note_string']] + df_ds['note_string'].tolist()
    vectorizer = TfidfVectorizer(
        tokenizer=lambda x: [t.strip() for t in x.replace(',', ' ').split()],
        token_pattern=None,
        min_df=1
    )
    tfidf_matrix = vectorizer.fit_transform(corpus)
    similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()

    df_ds = df_ds.copy()
    df_ds['similarity'] = similarities

    # penalizza note da evitare
    if profile['avoid_notes']:
        avoid_vec = vectorizer.transform([profile['avoid_notes']])
        avoid_sim = cosine_similarity(avoid_vec, tfidf_matrix[1:]).flatten()
        df_ds['similarity'] = df_ds['similarity'] - (avoid_sim * 0.3)

    # filtro qualità e moltiplicatore
    df_ds = _apply_quality(df_ds, quality_pct, niche)

    return _top_n(df_ds, n)


def get_similar_to(rid: str,
                   df_ratings: pd.DataFrame,
                   df_dataset: pd.DataFrame,
                   n: int = 8,
                   quality_pct: int = 40,
                   gender_filter='all',
                   known_idx: set | None = None,
                   niche: bool = False) -> pd.DataFrame:
    """
    Trova profumi simili a una voce specifica della lista (by stable id).
    Excludes every fragrance already in the list, not just the reference.
    """
    row = df_ratings[df_ratings['id'] == rid]

    if row.empty or not row.iloc[0].get('matched', False):
        return pd.DataFrame()

    if not build_note_string(row.iloc[0]):
        return pd.DataFrame()

    temp_profile_df = row.copy()
    temp_profile_df['matched'] = True
    # the reference is the target: never let a low rating turn it into
    # an "avoid" vector against itself
    temp_profile_df['rating'] = 10.0

    if known_idx is None:
        from matcher import known_dataset_indices
        known_idx = known_dataset_indices(df_ratings, df_dataset)

    return get_recommendations(
        temp_profile_df, df_dataset, n=n, exclude_known=True,
        gender_filter=gender_filter, quality_pct=quality_pct,
        known_idx=known_idx, niche=niche,
    )


def get_exploration_recommendations(df_ratings: pd.DataFrame,
                                     df_dataset: pd.DataFrame,
                                     explore_style: str,
                                     intensity: float = 0.5,
                                     n: int = 10,
                                     gender_filter='all',
                                     quality_pct: int = 40,
                                     known_idx: set | None = None,
                                     niche: bool = False) -> pd.DataFrame:

    profile = build_personal_profile(df_ratings)
    query = explore_style

    df_ds = prepare_dataset(df_dataset)

    # filtri
    df_ds = _filter_gender(df_ds, gender_filter)
    df_ds = _exclude_known(df_ds, df_ratings, df_dataset, known_idx)

    df_ds = df_ds[df_ds['note_string'].str.strip() != '']

    if df_ds.empty:
        return pd.DataFrame()

    # TF-IDF + cosine similarity — calcolato dopo tutti i filtri
    corpus = [query] + df_ds['note_string'].tolist()
    vectorizer = TfidfVectorizer(
        tokenizer=lambda x: [t.strip() for t in x.replace(',', ' ').split()],
        token_pattern=None,
        min_df=1
    )
    tfidf_matrix = vectorizer.fit_transform(corpus)
    similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()

    df_ds = df_ds.copy()
    df_ds['similarity'] = similarities

    # penalizza note da evitare
    if profile['avoid_notes']:
        avoid_vec = vectorizer.transform([profile['avoid_notes']])
        avoid_sim = cosine_similarity(avoid_vec, tfidf_matrix[1:]).flatten()
        df_ds['similarity'] = df_ds['similarity'] - (avoid_sim * 0.3)

    # salva profile_sim come colonna prima del filtro qualità
    if intensity < 1.0 and profile['note_string']:
        profile_vec = vectorizer.transform([profile['note_string']])
        profile_sim = cosine_similarity(profile_vec, tfidf_matrix[1:]).flatten()
        df_ds['profile_sim'] = profile_sim

    # filtro qualità e moltiplicatore
    df_ds = _apply_quality(df_ds, quality_pct, niche)

    # intensity — usa la colonna salvata prima del filtro
    if intensity < 1.0 and profile['note_string'] and 'profile_sim' in df_ds.columns:
        penalty_weight = (1.0 - intensity) * 0.5
        df_ds['similarity'] = df_ds['similarity'] - (
            (1 - df_ds['profile_sim']) * penalty_weight
        )
        df_ds = df_ds.drop(columns=['profile_sim'])

    return _top_n(df_ds, n)


def compute_scatter_data(df_ratings: pd.DataFrame) -> pd.DataFrame:

    FRESH_NOTES = {
        'citrus', 'bergamot', 'lemon', 'grapefruit', 'orange', 'lime',
        'mandarin', 'neroli', 'aquatic', 'marine', 'ozonic', 'water',
        'fresh', 'green', 'grass', 'fig', 'basil', 'mint', 'herbal',
        'lavender', 'rosemary', 'eucalyptus', 'petitgrain'
    }
    DEEP_NOTES = {
        'oud', 'agarwood', 'amber', 'ambergris', 'resin', 'resinous',
        'balsamic', 'benzoin', 'labdanum', 'incense', 'oriental',
        'leather', 'tobacco', 'smoky', 'smoke', 'tar', 'dark',
        'vanilla', 'tonka', 'caramel', 'gourmand', 'myrrh', 'frankincense',
        'licorice', 'coffee'
    }
    INTENSE_NOTES = {
        'oud', 'musk', 'ambergris', 'civet', 'castoreum', 'animalic',
        'amber', 'resin', 'tobacco', 'leather', 'vetiver', 'patchouli',
        'incense', 'myrrh', 'frankincense', 'benzoin', 'labdanum', 'coffee'
    }
    FRESH_ACCORDS = {
        'citrus', 'fresh', 'aquatic', 'green', 'aromatic', 'herbal',
        'fougere', 'fruity'
    }
    DEEP_ACCORDS = {
        'oriental', 'amber', 'oud', 'resinous', 'balsamic', 'leather',
        'tobacco', 'smoky', 'woody', 'warm spicy', 'gourmand', 'vanilla',
        'mossy', 'earthy', 'animalic', 'honey'
    }
    WEIGHTS = {'top': 0.3, 'middle': 0.6, 'base': 1.0}

    def score_notes(note_str, note_set, weight):
        tokens = set(str(note_str).lower().replace(',', ' ').split())
        return len(tokens & note_set) * weight

    matched_col = df_ratings.get('matched', pd.Series(False, index=df_ratings.index)).fillna(False).astype(bool)
    df_matched = df_ratings[matched_col].copy()

    if df_matched.empty:
        return pd.DataFrame()

    rows = []
    for _, row in df_matched.iterrows():
        top    = str(row.get('top_notes', ''))
        middle = str(row.get('middle_notes', ''))
        base   = str(row.get('base_notes', ''))

        freshness_raw = (
            score_notes(top,    FRESH_NOTES, WEIGHTS['top'])    +
            score_notes(middle, FRESH_NOTES, WEIGHTS['middle']) +
            score_notes(base,   FRESH_NOTES, WEIGHTS['base'])
        )
        depth_raw = (
            score_notes(top,    DEEP_NOTES, WEIGHTS['top'])    +
            score_notes(middle, DEEP_NOTES, WEIGHTS['middle']) +
            score_notes(base,   DEEP_NOTES, WEIGHTS['base'])
        )
        intensity_raw = (
            score_notes(top,    INTENSE_NOTES, WEIGHTS['top'])    +
            score_notes(middle, INTENSE_NOTES, WEIGHTS['middle']) +
            score_notes(base,   INTENSE_NOTES, WEIGHTS['base'])
        )

        # personal entries store accords as "woody, warm spicy, ..." in
        # main_accords (the old code read dataset-only mainaccord1..5 columns,
        # so accords were always ignored). Keep multi-word accords intact.
        accord_tokens = {
            a.strip().lower()
            for a in str(row.get('main_accords', '') or '').split(',')
            if a.strip() and a.strip().lower() != 'nan'
        }

        fresh_boost = 1.0 + 0.5 * len(accord_tokens & FRESH_ACCORDS)
        depth_boost = 1.0 + 0.5 * len(accord_tokens & DEEP_ACCORDS)
        fresh_base  = 1 if accord_tokens & FRESH_ACCORDS else 0
        depth_base  = 1 if accord_tokens & DEEP_ACCORDS else 0

        freshness = (freshness_raw + fresh_base) * fresh_boost
        depth     = (depth_raw + depth_base)     * depth_boost
        intensity = max(intensity_raw, 0.1)

        rows.append({
            'id':        row.get('id', ''),
            'name':      row.get('name', ''),
            'brand':     row.get('brand', ''),
            'form':      row.get('form', ''),
            'status':    row.get('status', ''),
            'rating':    row.get('rating', 5.0),
            'freshness': freshness,
            'depth':     depth,
            'intensity': intensity,
        })

    df = pd.DataFrame(rows)

    # trasformazione logaritmica per distribuire meglio i valori nello spazio
    # log1p(x) = log(1+x) — mantiene lo 0 a 0, amplifica i valori piccoli
    df['freshness'] = np.log1p(df['freshness'])
    df['depth']     = np.log1p(df['depth'])
    df['intensity'] = np.log1p(df['intensity'])

    # normalizzazione: 0 assoluto, max relativo alla collezione
    for col in ['freshness', 'depth', 'intensity']:
        col_max = df[col].max()
        df[col] = df[col] / col_max if col_max > 0 else df[col]

    df['intensity'] = df['intensity'].clip(lower=0.1)

    return df

def accord_scores(df_ratings: pd.DataFrame, accords: list[str],
                  liked_min: float = 7.5, disliked_max: float = 6.0) -> pd.DataFrame:
    """
    For each olfactory family, how present it is among liked vs disliked
    fragrances. Matching is on whole words, so 'rose' does not count
    'rosewood', 'amber' does not count 'ambergris', and 'spicy' is not
    double-counted inside 'warm spicy'.

    Returns columns: accord, liked, disliked, n_liked, n_disliked, all
    (liked/disliked normalized to the max across both, 0-1).
    """
    import re
    matched = df_ratings.get('matched', pd.Series(False, index=df_ratings.index))
    df = df_ratings[matched.fillna(False).astype(bool)]
    if df.empty:
        return pd.DataFrame(columns=['accord', 'liked', 'disliked',
                                     'n_liked', 'n_disliked', 'all'])

    # longest first, so 'warm spicy' consumes its words before 'spicy'
    ordered = sorted(accords, key=len, reverse=True)
    patterns = {a: re.compile(rf'(?<![a-z]){re.escape(a)}(?![a-z])') for a in ordered}

    def families(row) -> set:
        text = ' , '.join(str(row.get(c, '') or '') for c in
                          ['top_notes', 'middle_notes', 'base_notes', 'main_accords']).lower()
        found = set()
        for a in ordered:
            if patterns[a].search(text):
                found.add(a)
                text = patterns[a].sub(' ', text)
        return found

    fam = df.apply(families, axis=1)
    rows = []
    for a in accords:
        has = fam.map(lambda s: a in s)
        liked = df.loc[has & (df['rating'] >= liked_min), 'rating']
        disliked = df.loc[has & (df['rating'] <= disliked_max), 'rating']
        rows.append({
            'accord':     a,
            # weight by how much above/below the thresholds each rating is
            'liked':      float((liked - liked_min + 1).sum()),
            'disliked':   float((disliked_max - disliked + 1).sum()),
            'n_liked':    int(len(liked)),
            'n_disliked': int(len(disliked)),
            'all':        int(has.sum()),
        })
    out = pd.DataFrame(rows)
    top = max(out['liked'].max(), out['disliked'].max()) or 1
    out['liked'] /= top
    out['disliked'] /= top
    return out


def gap_analysis(df_ratings: pd.DataFrame, df_dataset: pd.DataFrame,
                 accords: list[str]) -> pd.DataFrame:
    """
    Compares how common each family is in the personal list vs the dataset.
    'gap' > 0 means the family is under-explored relative to the market.
    """
    mine = accord_scores(df_ratings, accords)
    n_mine = max(int(df_ratings.get('matched', pd.Series(dtype=bool)).fillna(False).sum()), 1)
    acc_cols = [f'mainaccord{i}' for i in range(1, 6)]
    ds_acc = df_dataset[acc_cols].fillna('').astype(str).apply(
        lambda r: {a.strip().lower() for a in r if a.strip()}, axis=1)
    n_ds = max(len(ds_acc), 1)
    rows = []
    for _, m in mine.iterrows():
        a = m['accord']
        market = ds_acc.map(lambda s: a in s).sum() / n_ds
        share = m['all'] / n_mine
        rows.append({'accord': a, 'tua_collezione': share, 'mercato': market,
                     'gap': market - share, 'n_tuoi': m['all']})
    return pd.DataFrame(rows).sort_values('gap', ascending=False)
