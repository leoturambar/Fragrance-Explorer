import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from matcher import load_dataset, get_candidates


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
        - avoid_notes: note associate a rating bassi (<= 5.5)

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

    # normalizza rating a 0-1, poi moltiplica per moltiplicatore status
    r_min, r_max = df['rating'].min(), df['rating'].max()
    df['_rating_w']     = (df['rating'] - r_min) / (r_max - r_min + 0.001)
    df['_status_mult']  = df.apply(_status_weight, axis=1)
    df['weight']        = df['_rating_w'] * df['_status_mult']

    weighted_parts = []
    for _, row in df.iterrows():
        note_str = build_note_string(row)
        if not note_str.strip():
            continue
        repeats = max(1, round(row['weight'] * 3))
        weighted_parts.extend([note_str] * repeats)

    df_top = df[df['rating'] >= 8.0]
    top_notes = ' '.join(build_note_string(r) for _, r in df_top.iterrows())

    df_avoid = df[df['rating'] <= 5.5]
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


def get_recommendations(df_ratings: pd.DataFrame,
                        df_dataset: pd.DataFrame,
                        n: int = 10,
                        exclude_known: bool = True,
                        gender_filter: str = 'all',
                        quality_pct: int = 40) -> pd.DataFrame:
    """
    Genera raccomandazioni dal dataset Kaggle basate sul profilo personale.
    """
    profile = build_personal_profile(df_ratings)

    if not profile['note_string']:
        return pd.DataFrame()

    df_ds = df_dataset.copy()

    # filtro qualità
    brand_scores = build_brand_scores(df_dataset)
    df_ds['quality_score'] = df_ds.apply(
        lambda r: compute_quality_score(r, brand_scores), axis=1
    )
    max_q = df_ds['quality_score'].max() or 1
    df_ds['quality_score'] = df_ds['quality_score'] / max_q

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

    # filtri
    if gender_filter != 'all':
        df_ds = df_ds[df_ds['Gender'].str.lower().str.contains(
            gender_filter, na=False)]

    if exclude_known:
        # Primary exclusion: confirmed matches via dataset_idx
        if 'dataset_idx' in df_ratings.columns:
            known_idxs = set(int(idx) for idx in df_ratings['dataset_idx'].dropna())
            df_ds = df_ds[~df_ds.index.isin(known_idxs)]
        # Secondary exclusion: brand+name match (covers unmatched personal entries)
        personal_keys = {
            (str(r['brand']).lower().strip(), str(r['name']).lower().strip())
            for _, r in df_ratings.iterrows()
        }
        df_ds = df_ds[~df_ds.apply(
            lambda r: (str(r['Brand']).lower().strip(),
                       str(r['Perfume']).lower().strip()) in personal_keys,
            axis=1
        )]

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
    quality_weight = 1 - (quality_pct / 100)
    min_threshold = quality_weight * 0.5
    df_ds = df_ds[df_ds['quality_score'] >= min_threshold]

    penalty = df_ds['quality_score'] ** (1 + quality_weight * 2)
    df_ds['similarity'] = df_ds['similarity'] * (
        (1 - quality_weight) + quality_weight * penalty
    )

    df_result = df_ds.nlargest(n, 'similarity')[
        ['Brand', 'Perfume', 'Gender', 'Top', 'Middle', 'Base',
         'mainaccord1', 'mainaccord2', 'similarity']
    ].copy()

    df_result['similarity'] = df_result['similarity'].round(3)
    df_result.columns = ['brand', 'name', 'gender', 'top', 'middle',
                         'base', 'accord1', 'accord2', 'similarity']

    df_result = df_result.drop_duplicates(subset=['brand', 'name'])
    return df_result.reset_index(drop=True)


def get_similar_to(brand: str, name: str,
                   df_ratings: pd.DataFrame,
                   df_dataset: pd.DataFrame,
                   n: int = 8,
                   quality_pct: int = 40) -> pd.DataFrame:
    """
    Trova profumi simili a uno specifico dalla tua lista.
    """
    row = df_ratings[
        (df_ratings['brand'].str.lower() == brand.lower()) &
        (df_ratings['name'].str.lower() == name.lower())
    ]

    if row.empty or not row.iloc[0].get('matched', False):
        return pd.DataFrame()

    note_str = build_note_string(row.iloc[0])
    if not note_str:
        return pd.DataFrame()

    temp_profile_df = row.copy()
    temp_profile_df['weight'] = 1.0
    temp_profile_df['matched'] = True

    return get_recommendations(
        temp_profile_df, df_dataset, n=n, exclude_known=True,
        quality_pct=quality_pct
    )


def get_exploration_recommendations(df_ratings: pd.DataFrame,
                                     df_dataset: pd.DataFrame,
                                     explore_style: str,
                                     intensity: float = 0.5,
                                     n: int = 10,
                                     gender_filter: str = 'all',
                                     quality_pct: int = 40) -> pd.DataFrame:

    profile = build_personal_profile(df_ratings)
    query = explore_style

    df_ds = df_dataset.copy()

    # filtro qualità
    brand_scores = build_brand_scores(df_dataset)
    df_ds['quality_score'] = df_ds.apply(
        lambda r: compute_quality_score(r, brand_scores), axis=1
    )
    max_q = df_ds['quality_score'].max() or 1
    df_ds['quality_score'] = df_ds['quality_score'] / max_q

    # costruisce stringa note
    df_ds['note_string'] = df_ds.apply(
        lambda r: ' '.join(filter(None, [
            str(r.get('Top', ''))    if pd.notna(r.get('Top'))    else '',
            str(r.get('Middle', '')) if pd.notna(r.get('Middle')) else '',
            str(r.get('Base', ''))   if pd.notna(r.get('Base'))   else '',
            ' '.join(str(r.get(f'mainaccord{i}', ''))
                     for i in range(1, 6)
                     if pd.notna(r.get(f'mainaccord{i}')))
        ])).lower(),
        axis=1
    )

    # filtri
    if gender_filter != 'all':
        df_ds = df_ds[df_ds['Gender'].str.lower().str.contains(
            gender_filter, na=False)]

    if 'dataset_idx' in df_ratings.columns:
        known_idxs = set(int(idx) for idx in df_ratings['dataset_idx'].dropna())
        df_ds = df_ds[~df_ds.index.isin(known_idxs)]
    personal_keys = {
        (str(r['brand']).lower().strip(), str(r['name']).lower().strip())
        for _, r in df_ratings.iterrows()
    }
    df_ds = df_ds[~df_ds.apply(
        lambda r: (str(r['Brand']).lower().strip(),
                   str(r['Perfume']).lower().strip()) in personal_keys,
        axis=1
    )]

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
    quality_weight = 1 - (quality_pct / 100)
    min_threshold = quality_weight * 0.5
    df_ds = df_ds[df_ds['quality_score'] >= min_threshold]

    penalty = df_ds['quality_score'] ** (1 + quality_weight * 2)
    df_ds['similarity'] = df_ds['similarity'] * (
        (1 - quality_weight) + quality_weight * penalty
    )

    # intensity — usa la colonna salvata prima del filtro
    if intensity < 1.0 and profile['note_string'] and 'profile_sim' in df_ds.columns:
        penalty_weight = (1.0 - intensity) * 0.5
        df_ds['similarity'] = df_ds['similarity'] - (
            (1 - df_ds['profile_sim']) * penalty_weight
        )
        df_ds = df_ds.drop(columns=['profile_sim'])

    df_result = df_ds.nlargest(n, 'similarity')[
        ['Brand', 'Perfume', 'Gender', 'Top', 'Middle', 'Base',
         'mainaccord1', 'mainaccord2', 'similarity']
    ].copy()

    df_result['similarity'] = df_result['similarity'].round(3)
    df_result.columns = ['brand', 'name', 'gender', 'top', 'middle',
                         'base', 'accord1', 'accord2', 'similarity']

    return df_result.reset_index(drop=True)


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

        accord_str = ' '.join(filter(None, [
            str(row.get('mainaccord1', '')),
            str(row.get('mainaccord2', '')),
            str(row.get('mainaccord3', '')),
            str(row.get('mainaccord4', '')),
            str(row.get('mainaccord5', '')),
        ])).lower()
        accord_tokens = set(accord_str.replace(',', ' ').split())

        fresh_boost = 1.0 + 0.5 * len(accord_tokens & FRESH_ACCORDS)
        depth_boost = 1.0 + 0.5 * len(accord_tokens & DEEP_ACCORDS)
        fresh_base  = 1 if accord_tokens & FRESH_ACCORDS else 0
        depth_base  = 1 if accord_tokens & DEEP_ACCORDS else 0

        freshness = (freshness_raw + fresh_base) * fresh_boost
        depth     = (depth_raw + depth_base)     * depth_boost
        intensity = max(intensity_raw, 0.1)

        rows.append({
            'name':      row.get('name', ''),
            'brand':     row.get('brand', ''),
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