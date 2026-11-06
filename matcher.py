import pandas as pd
import os
import re
import unicodedata
from rapidfuzz import process, fuzz

FRAGRANCES_CLEAN = 'data/fragrances_clean.csv'
FRAGRANCES_RAW   = 'data/fragrances_raw.csv'
LEGACY_MATCHES   = 'data/confirmed_matches.csv'

# Concentration / edition words stripped before comparing fragrance names
_CONCENTRATION_RE = re.compile(
    r'\b(eau de parfum|eau de toilette|eau de cologne|extrait de parfum|'
    r'extrait de cologne|extrait|parfum|edp|edt|edc|limited edition|'
    r'(?:19|20)\d\d)\b'
)

# Thresholds for deciding that a personal entry and a dataset row are the same
NAME_MATCH_MIN  = 90
KNOWN_NAME_MIN  = 95
BRAND_MATCH_MIN = 85
HEALTH_SUSPECT_BELOW = 75


# ── Dataset loading ───────────────────────────────────────────────────────────

def _parse_notes_from_description(desc: str) -> tuple[str, str, str]:
    """
    Extracts top, middle, base notes from the raw dataset description.
    E.g. "Top notes are Lemon, Rose; middle notes are Jasmine; base notes are Musk"
    """
    if not isinstance(desc, str):
        return '', '', ''
    top    = re.search(r'[Tt]op notes? (?:are|is) ([^;\.]+)', desc)
    middle = re.search(r'[Mm]iddle notes? (?:are|is) ([^;\.]+)', desc)
    base   = re.search(r'[Bb]ase notes? (?:are|is) ([^;\.]+)', desc)
    return (
        top.group(1).strip()    if top    else '',
        middle.group(1).strip() if middle else '',
        base.group(1).strip()   if base   else '',
    )


def _parse_accords(accord_str: str) -> list[str]:
    """Converts "['citrus', 'woody', 'musky']" into a clean list."""
    if not isinstance(accord_str, str):
        return []
    return re.findall(r"'([^']+)'", accord_str)


def _extract_brand_name_from_url(url: str) -> tuple[str, str]:
    """
    Extracts brand and name from a Fragrantica URL.
    https://www.fragrantica.com/perfume/Afnan/9am-70706.html → ('Afnan', '9am')
    """
    if not isinstance(url, str):
        return '', ''
    parts = url.rstrip('/').split('/')
    if len(parts) < 2:
        return '', ''
    brand = parts[-2].replace('-', ' ').title()
    name_raw = re.sub(r'\.html?$', '', parts[-1], flags=re.IGNORECASE)
    name_raw = re.sub(r'-\d+$', '', name_raw)
    return brand, name_raw.replace('-', ' ').title()


def _fix_mojibake(s):
    """Repairs strings decoded as latin-1/cp1252 that were actually UTF-8."""
    if not isinstance(s, str) or not re.search(r'[ÃÂâ]', s):
        return s
    try:
        return s.encode('cp1252').decode('utf-8')
    except (UnicodeEncodeError, UnicodeDecodeError):
        return s


def _read_csv_any(path: str, **kw) -> pd.DataFrame:
    try:
        return pd.read_csv(path, encoding='utf-8', **kw)
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding='cp1252', encoding_errors='replace', **kw)
        for col in df.select_dtypes(include=['object', 'string']).columns:
            df[col] = df[col].map(_fix_mojibake)
        return df


def _load_clean() -> pd.DataFrame:
    df = _read_csv_any(FRAGRANCES_CLEAN, sep=';')
    df.columns = df.columns.str.strip()
    df['Perfume'] = df['Perfume'].str.replace('-', ' ').str.title()
    df['Brand']   = df['Brand'].str.replace('-', ' ').str.title()
    df['Rating Value'] = df['Rating Value'].astype(str).str.replace(',', '.').astype(float)
    for col in ['Top', 'Middle', 'Base']:
        df[col] = df[col].str.title()
    df['source_dataset'] = 'clean'
    return df


def _load_raw() -> pd.DataFrame:
    df = _read_csv_any(FRAGRANCES_RAW)
    rows = []
    for row in df.to_dict('records'):
        brand, name = _extract_brand_name_from_url(row.get('url', ''))
        if not brand or not name:
            continue
        top, middle, base = _parse_notes_from_description(row.get('Description', ''))
        accords = _parse_accords(row.get('Main Accords', ''))
        accords += [''] * (5 - len(accords))
        rows.append({
            'Brand':        brand,
            'Perfume':      name,
            'Country':      '',
            'Gender':       str(row.get('Gender', '')),
            'Rating Value': float(str(row.get('Rating Value', 0)).replace(',', '.')),
            'Rating Count': row.get('Rating Count', 0),
            'Year':         '',
            'Top':    top.title()    if top    else '',
            'Middle': middle.title() if middle else '',
            'Base':   base.title()   if base   else '',
            'Perfumer1':    '',
            'Perfumer2':    '',
            'mainaccord1':  accords[0],
            'mainaccord2':  accords[1],
            'mainaccord3':  accords[2],
            'mainaccord4':  accords[3],
            'mainaccord5':  accords[4],
            'source_dataset': 'raw',
        })
    return pd.DataFrame(rows)


def _normalize_key(s: str) -> str:
    """Lowercase, strip accents and every non-alphanumeric character."""
    s = unicodedata.normalize('NFKD', str(s).lower().strip())
    s = s.encode('ascii', 'ignore').decode()
    return re.sub(r'[^a-z0-9]', '', s)


def _words(s: str) -> str:
    """Lowercase, accent-free, punctuation → spaces."""
    s = unicodedata.normalize('NFKD', str(s).lower())
    s = s.encode('ascii', 'ignore').decode()
    return re.sub(r'[^a-z0-9]+', ' ', s).strip()


def _clean_name(name: str, brand: str = '') -> str:
    """
    Name reduced to what identifies the fragrance: no brand words,
    no concentration words, no leading catalogue numbers ("02 L Air Du ...").
    """
    n = _words(name)
    n = _CONCENTRATION_RE.sub(' ', n)
    for w in _words(brand).split():
        if len(w) > 2:
            n = re.sub(rf'\b{re.escape(w)}\b', ' ', n)
    n = re.sub(r'^\d+\s+', '', n.strip())
    return re.sub(r'\s+', ' ', n).strip()


def load_dataset() -> pd.DataFrame:
    """
    Loads and merges the two datasets. The clean dataset wins on duplicates.
    Adds normalized helper columns used for matching.
    """
    df_clean = _load_clean()
    df_raw   = _load_raw()

    for df in [df_clean, df_raw]:
        df['_key'] = df['Brand'].map(_normalize_key) + '||' + df['Perfume'].map(_normalize_key)

    df_raw_only = df_raw[~df_raw['_key'].isin(set(df_clean['_key']))]
    df = pd.concat([df_clean, df_raw_only], ignore_index=True)
    df = df.drop_duplicates(subset=['_key']).reset_index(drop=True)

    df['brand_clean'] = df['Brand'].str.lower().str.strip()
    df['name_clean']  = df['Perfume'].str.lower().str.strip()
    df['brand_key']   = df['Brand'].map(_normalize_key)
    df['name_key']    = df['Perfume'].map(_normalize_key)
    df['gender_norm'] = df['Gender'].map(normalize_gender)
    return df.drop(columns=['_key'])


def normalize_gender(g) -> str:
    """Maps dataset gender labels to 'male' | 'female' | 'unisex' | ''."""
    g = str(g).lower().strip()
    if 'women and men' in g or 'unisex' in g:
        return 'unisex'
    if 'women' in g or 'female' in g:
        return 'female'
    if 'men' in g or 'male' in g:
        return 'male'
    return ''


# ── Matching helpers ──────────────────────────────────────────────────────────

_GENERIC_BRAND_WORDS = {
    'maison', 'parfums', 'parfum', 'perfumes', 'perfume', 'profumi', 'profumo',
    'les', 'des', 'house', 'paris', 'london', 'roma', 'milano', 'fragrances',
    'replica', 'atelier', 'eau', 'the',
}


def brands_match(a: str, b: str) -> bool:
    ka, kb = _normalize_key(a), _normalize_key(b)
    if not ka or not kb:
        return False
    if ka == kb or ka.startswith(kb) or kb.startswith(ka):
        return True
    if fuzz.ratio(ka, kb) >= BRAND_MATCH_MIN:
        return True
    # "Maison Margiela Replica" vs "Maison Martin Margiela": share a distinctive word
    wa = {w for w in _words(a).split() if len(w) >= 4} - _GENERIC_BRAND_WORDS
    wb = {w for w in _words(b).split() if len(w) >= 4} - _GENERIC_BRAND_WORDS
    return bool(wa & wb)


def name_similarity(user_clean: str, ds_clean: str) -> int:
    """
    Similarity between two cleaned names. When the personal name is a
    multi-word subset of the dataset name ("fico di amalfi" in
    "blu mediterraneo fico di amalfi") it counts as a full match; the reverse
    is not allowed, so "... marocain noir" never matches "... marocain".
    The personal name must appear as a contiguous phrase, so "terre d intense"
    does not match "terre d eau intense vetiver".
    """
    if not user_clean or not ds_clean:
        return 0
    score = fuzz.token_sort_ratio(user_clean, ds_clean)
    u, d = user_clean.split(), ds_clean.split()
    if len(u) >= 2 and len(u) < len(d) and f' {user_clean} ' in f' {ds_clean} ':
        score = max(score, 95)
    return round(score)


def match_score(brand: str, name: str, ds_brand: str, ds_name: str) -> int:
    """0-100 confidence that a personal entry and a dataset row are the same fragrance."""
    if not brands_match(brand, ds_brand):
        return 0
    return name_similarity(_clean_name(name, brand), _clean_name(ds_name, ds_brand))


_FORM_WORDS = {
    'edp': 'eau de parfum', 'edt': 'eau de toilette', 'edc': 'eau de cologne',
    'extrait': 'extrait', 'parfum': 'parfum',
}


def form_bonus(form: str, ds_name: str) -> int:
    """+1 when the dataset name mentions the same concentration as the entry."""
    words = _FORM_WORDS.get(str(form).lower().strip())
    return int(bool(words) and words in _words(ds_name))


def dataset_index_for(ds_brand: str, ds_name: str, df_dataset: pd.DataFrame):
    """Resolves a stored (brand, name) reference to the current dataset index."""
    if not ds_brand or not ds_name:
        return None
    hit = df_dataset.index[
        (df_dataset['brand_key'] == _normalize_key(ds_brand)) &
        (df_dataset['name_key'] == _normalize_key(ds_name))
    ]
    return int(hit[0]) if len(hit) else None


def resolve_dataset_indices(df_ratings: pd.DataFrame,
                            df_dataset: pd.DataFrame) -> pd.DataFrame:
    """Fills the derived 'dataset_idx' column from the stable ds_brand/ds_name keys."""
    df = df_ratings.copy()
    keys = df_dataset['brand_key'] + '||' + df_dataset['name_key']
    lookup = pd.Series(df_dataset.index, index=keys)
    lookup = lookup[~lookup.index.duplicated()]
    rk = df['ds_brand'].map(_normalize_key) + '||' + df['ds_name'].map(_normalize_key)
    df['dataset_idx'] = rk.map(lookup)
    df.loc[df['match_status'] != 'matched', 'dataset_idx'] = pd.NA
    return df


def dataset_notes(ds_row) -> dict:
    """Notes of a dataset row in the personal-list format."""
    def s(col):
        v = ds_row.get(col, '')
        return str(v) if pd.notna(v) and str(v) != 'nan' else ''
    acc = [s(f'mainaccord{i}') for i in range(1, 6)]
    return {
        'top':     s('Top'),
        'middle':  s('Middle'),
        'base':    s('Base'),
        'accords': ', '.join(a for a in acc if a),
    }


def get_candidates(brand: str, name: str, df_dataset: pd.DataFrame,
                   n: int = 5, threshold: int = 50) -> list[dict]:
    """Fuzzy candidates for a personal entry, best first."""
    # Work on distinct brands, then take *all* their rows (a row-level limit
    # used to cut off large brands such as Hermès or Diptyque)
    brands = df_dataset['brand_clean'].drop_duplicates()
    close = {m[0] for m in process.extract(brand.lower().strip(), brands,
                                           scorer=fuzz.ratio, limit=5)
             if m[1] >= max(threshold, 60)}
    close |= {b for b in brands if brands_match(brand, b)}
    if not close:
        return []

    df_brand = df_dataset[df_dataset['brand_clean'].isin(close)]
    target = _clean_name(name, brand)
    names = df_brand.apply(lambda r: _clean_name(r['Perfume'], r['Brand']), axis=1)
    scored = sorted(
        ((name_similarity(target, nm), ds_idx) for ds_idx, nm in names.items()),
        key=lambda t: t[0], reverse=True
    )[:n]

    candidates = []
    for score, ds_idx in scored:
        row = df_dataset.loc[ds_idx]
        notes = dataset_notes(row)
        candidates.append({
            'dataset_idx': int(ds_idx),
            'brand':       row['Brand'],
            'name':        row['Perfume'],
            'score':       round(score),
            'top':         notes['top'],
            'middle':      notes['middle'],
            'base':        notes['base'],
            'accords':     notes['accords'],
        })
    return sorted(candidates, key=lambda x: x['score'], reverse=True)


def known_dataset_indices(df_ratings: pd.DataFrame,
                          df_dataset: pd.DataFrame) -> set[int]:
    """
    Dataset rows that correspond to fragrances already in the personal list:
    confirmed matches plus fuzzy brand+name matches (catches brand spelling
    differences such as "Memo" vs "Memo Paris" and unmatched entries).
    """
    known = set()
    if 'dataset_idx' in df_ratings.columns:
        known |= {int(i) for i in df_ratings['dataset_idx'].dropna()}

    for _, r in df_ratings.iterrows():
        bkey = _normalize_key(r['brand'])
        if not bkey:
            continue
        same_brand = df_dataset[
            (df_dataset['brand_key'] == bkey) |
            df_dataset['brand_key'].str.startswith(bkey) |
            df_dataset['brand_key'].map(lambda k: bool(k) and bkey.startswith(k))
        ]
        if same_brand.empty:
            # fall back to fuzzy brand comparison (e.g. "Comme de Garcons")
            brands = df_dataset['brand_key'].drop_duplicates()
            close = [b for b in brands if fuzz.ratio(bkey, b) >= BRAND_MATCH_MIN]
            same_brand = df_dataset[df_dataset['brand_key'].isin(close)]
        target = _clean_name(r['name'], r['brand'])
        if not target:
            continue
        # strict comparison: flankers ("Encre Noire Sport", "Wulong Cha X")
        # are different fragrances and must stay recommendable
        for ds_idx, ds in same_brand.iterrows():
            ds_clean = _clean_name(ds['Perfume'], ds['Brand'])
            if fuzz.token_sort_ratio(target, ds_clean) >= KNOWN_NAME_MIN:
                known.add(int(ds_idx))
    return known


def match_health(df_ratings: pd.DataFrame, df_dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Flags entries whose data looks wrong:
      - matched to a dataset row whose brand/name doesn't resemble the entry
      - dataset reference not found in the current dataset
      - notes_source 'dataset' but notes differ from the referenced row
      - no notes at all
    """
    issues = []
    for _, r in df_ratings.iterrows():
        rid, lbl = r['id'], f"{r['brand']} — {r['name']} {r['form']}".strip()
        if r['match_status'] == 'matched':
            ds_idx = r.get('dataset_idx')
            if pd.isna(ds_idx):
                issues.append({'id': rid, 'profumo': lbl, 'gravità': 'alta',
                               'problema': f"Abbinamento a '{r['ds_brand']} — {r['ds_name']}' "
                                           "non trovato nel dataset attuale"})
            else:
                score = match_score(r['brand'], r['name'], r['ds_brand'], r['ds_name'])
                if score < HEALTH_SUSPECT_BELOW:
                    issues.append({'id': rid, 'profumo': lbl, 'gravità': 'alta',
                                   'problema': f"Abbinato a '{r['ds_brand']} — {r['ds_name']}' "
                                               f"(somiglianza {score}%)"})
                if r['notes_source'] == 'dataset':
                    ds_top = dataset_notes(df_dataset.loc[int(ds_idx)])['top']
                    if ds_top.lower() != str(r['top_notes']).lower():
                        issues.append({'id': rid, 'profumo': lbl, 'gravità': 'media',
                                       'problema': "Le note salvate non coincidono con quelle "
                                                   "del profumo abbinato"})
        if not r['matched']:
            issues.append({'id': rid, 'profumo': lbl, 'gravità': 'media',
                           'problema': 'Nessuna nota olfattiva — escluso da profilo e grafici'})
        elif not any(str(r[c]).strip() for c in ('top_notes', 'middle_notes', 'base_notes')):
            issues.append({'id': rid, 'profumo': lbl, 'gravità': 'bassa',
                           'problema': 'Solo accordi, nessuna piramide — '
                                       'incolla le note in Arricchisci note'})
    return pd.DataFrame(issues, columns=['id', 'profumo', 'gravità', 'problema'])


# ── Legacy support ────────────────────────────────────────────────────────────

def load_legacy_matches(path: str = LEGACY_MATCHES) -> dict:
    """
    Reads the old positional confirmed_matches.csv (rating row position →
    dataset index). Only used by the one-off migration in repair_data.py.
    """
    if not os.path.exists(path):
        return {}
    df = pd.read_csv(path)
    out = {}
    for _, row in df.iterrows():
        out[int(row['rating_idx'])] = (
            None if pd.isna(row['dataset_idx']) else int(row['dataset_idx'])
        )
    return out
