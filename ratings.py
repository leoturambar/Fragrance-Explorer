from bs4 import BeautifulSoup
import pandas as pd
import os
import uuid
from datetime import date

MY_LIST_FILE = 'data/my_list.html'
MY_RATINGS_FILE = 'data/my_ratings.csv'

RATINGS_COLS = [
    'id',                # stable 8-char hex id — never changes, used as key everywhere
    'brand',
    'name',
    'form',              # EdP, EdT, Extrait, ...
    'status',            # Bottle | Decant | Sample | Tested | '' (none)
    'watchlist',         # True/False — bookmark independent from status
    'rating',            # 1.0 → 10.0
    'bottle_candidate',  # True/False — N/A if status == Bottle
    'comment',           # personal comment
    'source',            # html_import, manual
    'date_added',        # ISO date (YYYY-MM-DD), '' for legacy entries
]

# Columns describing olfactory notes and the dataset match
NOTE_COLS = [
    'top_notes', 'middle_notes', 'base_notes', 'main_accords',
    'match_status',      # 'matched' | 'skipped' | '' (pending)
    'ds_brand',          # brand of the matched dataset row (stable key)
    'ds_name',           # name of the matched dataset row (stable key)
    'notes_source',      # 'dataset' | 'web_enriched' | ''
    'notes_url',         # Fragrantica URL when web_enriched
]

STRING_COLS = ['id', 'brand', 'name', 'form', 'status', 'comment', 'source',
               'date_added'] + NOTE_COLS
BOOL_COLS = ['watchlist', 'bottle_candidate']

# Columns that are derived at load time and never persisted
DERIVED_COLS = ['matched', 'dataset_idx']

STATUS_VALUES = ['', 'Bottle', 'Decant', 'Sample', 'Tested']

_OWNERSHIP_TO_STATUS = {
    'Full Bottle': 'Bottle',
    'Decant':      'Decant',
    'Sample':      'Sample',
    'Wishlist':    '',
}


def new_id() -> str:
    return uuid.uuid4().hex[:8]


def _migrate_ownership(df: pd.DataFrame) -> pd.DataFrame:
    """Migrates old 'ownership' column to 'status' + 'watchlist'."""
    if 'ownership' not in df.columns:
        return df
    own = df['ownership'].fillna('').astype(str).str.strip()
    if 'status' not in df.columns:
        df['status'] = own.map(_OWNERSHIP_TO_STATUS).fillna('')
    if 'watchlist' not in df.columns:
        df['watchlist'] = own.str.lower() == 'wishlist'
    return df.drop(columns=['ownership'])


def _to_bool(s: pd.Series) -> pd.Series:
    return s.map(lambda v: str(v).strip().lower() in ('true', '1', 'yes')
                 if pd.notna(v) else False)


def _normalize_schema(df: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    """
    Ensures every expected column exists with a clean dtype.
    Returns (df, changed) — changed=True when the file needs to be rewritten.
    """
    changed = False
    for col in RATINGS_COLS + NOTE_COLS:
        if col not in df.columns:
            df[col] = ''
            changed = True
    for col in STRING_COLS:
        df[col] = df[col].fillna('').astype(str)
        df.loc[df[col] == 'nan', col] = ''
    for col in BOOL_COLS:
        df[col] = _to_bool(df[col])
    # NaN is allowed: watchlist-only entries (not yet smelled) have no rating
    df['rating'] = pd.to_numeric(df['rating'], errors='coerce')

    missing_id = df['id'].str.strip() == ''
    if missing_id.any():
        df.loc[missing_id, 'id'] = [new_id() for _ in range(missing_id.sum())]
        changed = True

    # Legacy HTML import put "(Decant)" into 'form' when there was no form span
    bad_form = df['form'].str.match(r'^\(.*\)$')
    if bad_form.any():
        for i in df.index[bad_form]:
            own = df.at[i, 'form'].strip('()')
            if not df.at[i, 'status']:
                df.at[i, 'status'] = _OWNERSHIP_TO_STATUS.get(own, '')
            df.at[i, 'form'] = ''
        changed = True

    return df, changed


def _add_derived(df: pd.DataFrame) -> pd.DataFrame:
    """'matched' = the entry has usable olfactory notes."""
    has_notes = (
        (df['top_notes'].str.strip() != '') |
        (df['middle_notes'].str.strip() != '') |
        (df['base_notes'].str.strip() != '') |
        (df['main_accords'].str.strip() != '')
    )
    df['matched'] = has_notes
    if 'dataset_idx' not in df.columns:
        df['dataset_idx'] = pd.NA
    return df


def parse_html() -> pd.DataFrame:
    """
    Reads my_list.html and extracts every fragrance with rating and comment.
    Returns a DataFrame with the legacy 'ownership' column (migrated later).
    """
    if not os.path.exists(MY_LIST_FILE):
        return pd.DataFrame(columns=RATINGS_COLS)

    with open(MY_LIST_FILE, 'r', encoding='utf-8') as f:
        soup = BeautifulSoup(f.read(), 'html.parser')

    rows = []
    for section in soup.find_all('section', class_='section'):
        score_el = section.find(class_='score')
        if not score_el:
            continue
        rating_str = score_el.get_text(strip=True).replace('/10', '')
        try:
            rating = float(rating_str)
        except ValueError:
            continue

        for item in section.find_all('div', class_='item'):
            brand_el = item.find('span', class_='brand')
            brand = brand_el.get_text(strip=True) if brand_el else ''
            name_el = item.find('span', class_='name')
            name = name_el.get_text(strip=True) if name_el else ''

            # span.form can hold either the concentration ("EdP") or the
            # ownership in parentheses ("(Decant)") — classify by shape
            form, ownership = '', ''
            for el in item.find_all('span', class_='form'):
                txt = el.get_text(strip=True)
                if txt.startswith('(') and txt.endswith(')'):
                    ownership = txt.strip('()')
                elif not form:
                    form = txt

            bottle_candidate = bool(item.find('span', class_='bc'))
            desc_el = item.find('div', class_='desc')
            comment = desc_el.get_text(strip=True) if desc_el else ''

            rows.append({
                'brand':            brand,
                'name':             name,
                'form':             form,
                'ownership':        ownership,
                'rating':           rating,
                'bottle_candidate': bottle_candidate,
                'comment':          comment,
                'source':           'html_import',
            })

    return pd.DataFrame(rows)


def load_ratings() -> pd.DataFrame:
    """
    Loads the ratings CSV (creating it from the HTML export if missing),
    migrates old schemas and adds derived columns.
    """
    if not os.path.exists(MY_RATINGS_FILE):
        df = parse_html()
        if df.empty:
            df = pd.DataFrame(columns=RATINGS_COLS + NOTE_COLS)
            df, _ = _normalize_schema(df)
            return _add_derived(df)
        df = _migrate_ownership(df)
        df, _ = _normalize_schema(df)
        save_ratings(df)
        return _add_derived(df)

    df = pd.read_csv(MY_RATINGS_FILE, encoding='utf-8')
    changed = 'ownership' in df.columns
    df = _migrate_ownership(df)
    df, schema_changed = _normalize_schema(df)
    if changed or schema_changed:
        save_ratings(df)
    return _add_derived(df)


def save_ratings(df: pd.DataFrame):
    """Persists the ratings, dropping derived columns."""
    cols = RATINGS_COLS + NOTE_COLS
    extra = [c for c in df.columns if c not in cols and c not in DERIVED_COLS]
    df[cols + extra].to_csv(MY_RATINGS_FILE, index=False, encoding='utf-8')


def _row_index(df: pd.DataFrame, rid: str):
    hits = df.index[df['id'] == rid]
    if len(hits) == 0:
        raise KeyError(f"Rating id {rid} not found")
    return hits[0]


def get_by_id(df: pd.DataFrame, rid: str) -> pd.Series:
    return df.loc[_row_index(df, rid)]


def update_rating(rid: str, **fields) -> pd.DataFrame:
    """Updates arbitrary fields of one entry, addressed by its stable id."""
    df = load_ratings()
    i = _row_index(df, rid)
    for k, v in fields.items():
        df.at[i, k] = v
    save_ratings(df)
    return _add_derived(df)


def set_notes(rid: str, top: str, middle: str, base: str, accords: str,
              source: str, url: str = '',
              ds_brand: str | None = None, ds_name: str | None = None,
              match_status: str | None = None) -> pd.DataFrame:
    """Writes olfactory notes (and optionally the dataset match) for one entry."""
    fields = {
        'top_notes': top or '', 'middle_notes': middle or '',
        'base_notes': base or '', 'main_accords': accords or '',
        'notes_source': source, 'notes_url': url or '',
    }
    if ds_brand is not None:
        fields['ds_brand'] = ds_brand
    if ds_name is not None:
        fields['ds_name'] = ds_name
    if match_status is not None:
        fields['match_status'] = match_status
    return update_rating(rid, **fields)


def save_enriched_notes(rid: str, top: str, middle: str, base: str,
                        accords: str, url: str = '') -> pd.DataFrame:
    """Stores notes scraped from Fragrantica. Leaves the dataset match alone."""
    return set_notes(rid, top, middle, base, accords,
                     source='web_enriched', url=url)


def bulk_update(changes: dict[str, dict]) -> pd.DataFrame:
    """Applies {id: {field: value}} in a single load/save."""
    df = load_ratings()
    for rid, fields in changes.items():
        i = _row_index(df, rid)
        for k, v in fields.items():
            df.at[i, k] = v
    df.loc[df['status'] == 'Bottle', 'bottle_candidate'] = False
    save_ratings(df)
    return _add_derived(df)


def is_watchlist_only(row) -> bool:
    """Bookmarked but never tried: no status, watchlist on."""
    return not str(row.get('status', '') or '').strip() and bool(row.get('watchlist', False))


def tried(df: pd.DataFrame) -> pd.DataFrame:
    """Entries that have actually been smelled (excludes watchlist-only bookmarks)."""
    wl_only = (df['status'].fillna('').str.strip() == '') & df['watchlist'].astype(bool)
    return df[~wl_only & df['rating'].notna()]


def add_rating(brand: str, name: str, form: str, status: str,
               rating: float | None, comment: str = '',
               bottle_candidate: bool = False, watchlist: bool = False,
               notes: dict | None = None) -> str:
    """
    Adds a new fragrance and returns its id.
    rating=None is allowed for watchlist-only entries.
    notes: optional {top, middle, base, accords, ds_brand, ds_name, source}.
    """
    df = load_ratings()
    rid = new_id()
    new_row = {c: '' for c in RATINGS_COLS + NOTE_COLS}
    new_row.update({
        'id':               rid,
        'brand':            brand.strip(),
        'name':             name.strip(),
        'form':             form,
        'status':           status,
        'watchlist':        bool(watchlist),
        'rating':           float(rating) if rating is not None else float('nan'),
        'bottle_candidate': bool(bottle_candidate) and status != 'Bottle',
        'comment':          comment,
        'source':           'manual',
        'date_added':       date.today().isoformat(),
    })
    if notes:
        new_row.update({
            'top_notes':    notes.get('top', ''),
            'middle_notes': notes.get('middle', ''),
            'base_notes':   notes.get('base', ''),
            'main_accords': notes.get('accords', ''),
            'notes_source': notes.get('source', 'dataset'),
            'ds_brand':     notes.get('ds_brand', ''),
            'ds_name':      notes.get('ds_name', ''),
            'match_status': 'matched' if notes.get('ds_brand') else '',
        })
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    save_ratings(df)
    return rid


def delete_rating(rid: str):
    """Removes one entry by id. Other entries keep their ids and matches."""
    df = load_ratings()
    df = df[df['id'] != rid].reset_index(drop=True)
    save_ratings(df)


def check_duplicate(brand: str, name: str, form: str = '') -> bool:
    """True if an entry with the same brand, name and form already exists."""
    df = load_ratings()
    if df.empty:
        return False
    same = (
        (df['brand'].str.lower().str.strip() == brand.lower().strip()) &
        (df['name'].str.lower().str.strip() == name.lower().strip())
    )
    if form:
        same &= df['form'].str.lower().str.strip() == form.lower().strip()
    return bool(same.any())


def label(row) -> str:
    """Human label that disambiguates entries with same brand+name."""
    form = f" {row['form']}" if str(row.get('form', '')).strip() else ''
    return f"{row['brand']} — {row['name']}{form}"


def apply_match(rid: str, ds_brand: str, ds_name: str, ds_notes: dict) -> pd.DataFrame:
    """
    Links one entry to a dataset row. Notes are taken from the dataset when it
    has a pyramid; otherwise existing notes (e.g. from Fragrantica) are kept
    and only the reference is stored, so a match never wipes better data.
    """
    df = load_ratings()
    i = _row_index(df, rid)
    ds_has_pyramid = any(ds_notes.get(k) for k in ('top', 'middle', 'base'))
    has_notes = any(str(df.at[i, c]).strip() for c in
                    ('top_notes', 'middle_notes', 'base_notes', 'main_accords'))
    if ds_has_pyramid or not has_notes:
        df.at[i, 'top_notes']    = ds_notes.get('top', '')
        df.at[i, 'middle_notes'] = ds_notes.get('middle', '')
        df.at[i, 'base_notes']   = ds_notes.get('base', '')
        df.at[i, 'main_accords'] = ds_notes.get('accords', '')
        df.at[i, 'notes_source'] = 'dataset'
        df.at[i, 'notes_url']    = ''
    elif not str(df.at[i, 'main_accords']).strip():
        df.at[i, 'main_accords'] = ds_notes.get('accords', '')
    df.at[i, 'ds_brand'] = ds_brand
    df.at[i, 'ds_name'] = ds_name
    df.at[i, 'match_status'] = 'matched'
    save_ratings(df)
    return _add_derived(df)


def set_match_status(rid: str, status: str) -> pd.DataFrame:
    """status: 'skipped' (not in dataset) or '' (back to pending). Notes are kept."""
    return update_rating(rid, match_status=status, ds_brand='', ds_name='')
