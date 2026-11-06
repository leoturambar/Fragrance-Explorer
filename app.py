import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import date

from config import ACCORDS, EXPLORATION_STYLES, GENDER_OPTIONS, MIN_RATING, MAX_RATING, RATING_STEP
from enricher import enrich_from_web, scrape_fragrantica
from ratings import (load_ratings, add_rating, delete_rating, check_duplicate,
                     save_enriched_notes, set_notes, bulk_update, apply_match,
                     set_match_status, tried, is_watchlist_only, label,
                     STATUS_VALUES, MY_RATINGS_FILE)
from matcher import (load_dataset, get_candidates, dataset_notes, resolve_dataset_indices,
                     known_dataset_indices, match_health)
from recommender import (get_recommendations, get_similar_to, get_exploration_recommendations,
                         compute_scatter_data, prepare_dataset, accord_scores, gap_analysis)
from llm import explain_recommendation, explain_profile, explain_exploration

# ── Setup ─────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Fragrance Explorer", page_icon="🌿", layout="wide")


@st.cache_resource(show_spinner="Carico il dataset…")
def get_dataset() -> pd.DataFrame:
    return load_dataset()


@st.cache_resource(show_spinner="Preparo il dataset per le raccomandazioni…")
def get_prepared_dataset() -> pd.DataFrame:
    return prepare_dataset(get_dataset())


@st.cache_data(show_spinner=False)
def get_known_idx(keys: tuple) -> set:
    """keys: tuple of (brand, name, dataset_idx) — cache key for the exclusion set."""
    df = pd.DataFrame(list(keys), columns=['brand', 'name', 'dataset_idx'])
    return known_dataset_indices(df, get_dataset())


@st.cache_data(show_spinner=False)
def get_candidates_cached(brand: str, name: str, n: int = 10) -> list[dict]:
    return get_candidates(brand, name, get_dataset(), n=n, threshold=35)


def get_ratings() -> pd.DataFrame:
    return resolve_dataset_indices(load_ratings(), get_dataset())


def known_idx_for(df: pd.DataFrame) -> set:
    keys = tuple(
        (r['brand'], r['name'], None if pd.isna(r['dataset_idx']) else int(r['dataset_idx']))
        for _, r in df.iterrows()
    )
    return get_known_idx(keys)


df_dataset = get_dataset()
df_ratings = get_ratings()

RATING_OPTIONS = [r / 10 for r in range(int(MIN_RATING * 10), int(MAX_RATING * 10) + 1,
                                        int(RATING_STEP * 10))]
STATUS_ICONS = {'Bottle': '🫙', 'Decant': '🧪', 'Sample': '🧴', 'Tested': '👃'}
STATUS_HELP = ("Bottle = possiedi la bottiglia · Decant/Sample = formato ridotto · "
               "Tested = provato, non posseduto · Nessuno = non ancora provato")
SCATTER_OPTS = {
    'all':            'Tutti i provati',
    'tried_or_owned': 'Con status (Bottle+Decant+Sample+Tested)',
    'owned':          'Posseduti (Bottle+Decant+Sample)',
    'bottle_only':    'Solo bottiglie',
}
FORMS = ["EdP", "EdT", "Extrait", "Parfum", "EdC", "Cologne Intense", ""]


def fmt_rating(v) -> str:
    return '—' if pd.isna(v) else f"{v:g}"


def status_icons(row) -> str:
    s = str(row.get('status', '') or '')
    return (STATUS_ICONS.get(s, '')
            + ('⭐' if row.get('bottle_candidate') and s != 'Bottle' else '')
            + ('👁' if row.get('watchlist') else ''))


def notes_provenance(row) -> str:
    src = row.get('notes_source', '')
    if not row.get('matched'):
        return "⚠️ Nessuna nota olfattiva"
    pyramid = any(str(row.get(c, '')).strip() for c in ('top_notes', 'middle_notes', 'base_notes'))
    tail = '' if pyramid else ' · solo accordi'
    if src == 'web_enriched':
        url = row.get('notes_url', '')
        return f"🌐 Note da [Fragrantica]({url}){tail}" if url else f"🌐 Note da Fragrantica{tail}"
    if row.get('match_status') == 'matched':
        return f"📚 Dataset: {row['ds_brand']} — {row['ds_name']}{tail}"
    return f"📝 Note inserite{tail}"


def show_notes(top, middle, base, accords=None):
    for lbl, v in (("Top", top), ("Cuore", middle), ("Fondo", base), ("Accordi", accords)):
        if v is not None and pd.notna(v) and str(v).strip():
            st.caption(f"**{lbl}:** {v}")


tab_lista, tab_profilo, tab_scopri, tab_gestione = st.tabs([
    "📋 La mia lista", "🧬 Profilo", "✨ Scopri", "⚙️ Gestione",
])

# ═══════════════════════════════════════════════════════════════════════════════
# Tab 1 — La mia lista
# ═══════════════════════════════════════════════════════════════════════════════

with tab_lista:
    st.header("La mia lista")

    if df_ratings.empty:
        st.info("Nessuna fragranza ancora. Usa il tab Gestione per aggiungerne.")
    else:
        c_search, c_view = st.columns([3, 1])
        with c_search:
            query = st.text_input("🔎 Cerca", placeholder="brand, nome, nota, commento…",
                                  key="list_search")
        with c_view:
            view = st.segmented_control("Vista", ["Schede", "Tabella"], default="Schede",
                                        key="list_view") or "Schede"

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            r_lo, r_hi = st.slider("Rating", MIN_RATING, MAX_RATING,
                                   (MIN_RATING, MAX_RATING), RATING_STEP, key="list_rating")
        with col2:
            status_filter = st.selectbox(
                "Status", ["Tutti", "Bottle", "Decant", "Sample", "Tested",
                           "Watchlist", "Nessuno status"], key="list_status")
        with col3:
            only_bc = st.checkbox("Solo bottle candidates", key="list_bc")
            only_issues = st.checkbox("Solo senza piramide", key="list_issues",
                                      help="Profumi senza note top/cuore/fondo")
        with col4:
            sort_by = st.selectbox("Ordina per", ["Rating ↓", "Rating ↑", "Brand A-Z",
                                                  "Aggiunti di recente"], key="list_sort")

        df_view = df_ratings.copy()
        if query:
            q = query.lower()
            hay = df_view[['brand', 'name', 'form', 'comment', 'top_notes', 'middle_notes',
                           'base_notes', 'main_accords']].astype(str).agg(' '.join, axis=1).str.lower()
            df_view = df_view[hay.str.contains(q, regex=False)]
        if status_filter == "Watchlist":
            df_view = df_view[df_view['watchlist']]
        elif status_filter == "Nessuno status":
            df_view = df_view[df_view['status'] == '']
        elif status_filter != "Tutti":
            df_view = df_view[df_view['status'] == status_filter]
        if only_bc:
            df_view = df_view[df_view['bottle_candidate']]
        if only_issues:
            no_pyr = (df_view[['top_notes', 'middle_notes', 'base_notes']]
                      .apply(lambda c: c.str.strip() == '').all(axis=1))
            df_view = df_view[no_pyr]
        in_range = df_view['rating'].between(r_lo, r_hi)
        if (r_lo, r_hi) != (MIN_RATING, MAX_RATING):
            df_view = df_view[in_range]
        else:
            df_view = df_view[in_range | df_view['rating'].isna()]

        if sort_by == "Rating ↓":
            df_view = df_view.sort_values('rating', ascending=False, na_position='last')
        elif sort_by == "Rating ↑":
            df_view = df_view.sort_values('rating', ascending=True, na_position='last')
        elif sort_by == "Brand A-Z":
            df_view = df_view.sort_values(['brand', 'name'], key=lambda s: s.str.lower())
        else:
            df_view = df_view.sort_values('date_added', ascending=False)

        st.caption(f"{len(df_view)} di {len(df_ratings)} fragranze")

        # ── Vista tabella (modifica rapida in blocco) ──────────────────────────
        if view == "Tabella":
            edit_cols = ['brand', 'name', 'form', 'status', 'rating', 'watchlist',
                         'bottle_candidate', 'comment']
            df_table = df_view.set_index('id')[edit_cols + ['main_accords']]
            edited = st.data_editor(
                df_table,
                hide_index=True,
                width='stretch',
                height=min(700, 40 + 35 * len(df_table)),
                disabled=['main_accords'],
                column_config={
                    'brand':   st.column_config.TextColumn("Brand", required=True),
                    'name':    st.column_config.TextColumn("Nome", required=True),
                    'form':    st.column_config.SelectboxColumn("Formato", options=FORMS),
                    'status':  st.column_config.SelectboxColumn("Status", options=STATUS_VALUES,
                                                                help=STATUS_HELP),
                    'rating':  st.column_config.NumberColumn("Rating", min_value=MIN_RATING,
                                                             max_value=MAX_RATING,
                                                             step=RATING_STEP, format="%.1f"),
                    'watchlist':        st.column_config.CheckboxColumn("👁"),
                    'bottle_candidate': st.column_config.CheckboxColumn("⭐ BC"),
                    'comment':          st.column_config.TextColumn("Commento", width='large'),
                    'main_accords':     st.column_config.TextColumn("Accordi"),
                },
                key="table_editor",
            )
            changes = {}
            for rid in df_table.index:
                diff = {c: edited.at[rid, c] for c in edit_cols
                        if not (pd.isna(edited.at[rid, c]) and pd.isna(df_table.at[rid, c]))
                        and edited.at[rid, c] != df_table.at[rid, c]}
                if diff:
                    changes[rid] = diff
            if changes:
                st.info(f"{len(changes)} voci modificate.")
                if st.button("💾 Salva modifiche", type="primary", key="table_save"):
                    bulk_update(changes)
                    st.session_state.pop('table_editor', None)
                    st.toast("Modifiche salvate")
                    st.rerun()

        # ── Vista schede ───────────────────────────────────────────────────────
        else:
            for _, row in df_view.iterrows():
                rid = row['id']
                icons = status_icons(row)
                warn = '' if row['matched'] else ' ⚠️'
                with st.expander(
                    f"**{fmt_rating(row['rating'])}** — {row['brand']} {row['name']}"
                    + (f" · {row['form']}" if row['form'] else '')
                    + (f"  {icons}" if icons else '') + warn
                ):
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        if row['comment']:
                            st.write(row['comment'])
                        show_notes(row['top_notes'], row['middle_notes'], row['base_notes'],
                                   row['main_accords'])
                    with col2:
                        st.caption(f"Status: {row['status'] or '—'}"
                                   + (" · 👁 Watchlist" if row['watchlist'] else "")
                                   + (" · ⭐ Bottle candidate"
                                      if row['bottle_candidate'] and row['status'] != 'Bottle' else ""))
                        if row['date_added']:
                            st.caption(f"Aggiunto il {row['date_added']}")
                        st.caption(notes_provenance(row))

                        col_edit, col_del = st.columns(2)
                        with col_edit:
                            if st.button("✏️ Modifica", key=f"edit_{rid}"):
                                st.session_state[f"editing_{rid}"] = True
                        with col_del:
                            if not st.session_state.get(f"confirm_delete_{rid}"):
                                if st.button("🗑️ Elimina", key=f"del_{rid}"):
                                    st.session_state[f"confirm_delete_{rid}"] = True
                                    st.rerun()
                            else:
                                st.warning("Sicuro?")
                                col_yes, col_no = st.columns(2)
                                with col_yes:
                                    if st.button("Sì", key=f"del_confirm_{rid}", type="primary"):
                                        delete_rating(rid)
                                        st.session_state.pop(f"confirm_delete_{rid}", None)
                                        st.toast(f"Eliminato {row['brand']} {row['name']}")
                                        st.rerun()
                                with col_no:
                                    if st.button("No", key=f"del_cancel_{rid}"):
                                        st.session_state.pop(f"confirm_delete_{rid}", None)
                                        st.rerun()

                    if st.session_state.get(f"editing_{rid}"):
                        with st.form(key=f"form_edit_{rid}"):
                            fc1, fc2, fc3 = st.columns([2, 2, 1])
                            e_brand = fc1.text_input("Brand", value=row['brand'])
                            e_name = fc2.text_input("Nome", value=row['name'])
                            e_form = fc3.selectbox(
                                "Formato", FORMS,
                                index=FORMS.index(row['form']) if row['form'] in FORMS else len(FORMS) - 1)
                            e_rating = st.select_slider(
                                "Rating", options=RATING_OPTIONS,
                                value=float(row['rating']) if pd.notna(row['rating']) else 7.0)
                            e_comment = st.text_area("Commento", value=row['comment'])
                            sc1, sc2, sc3 = st.columns(3)
                            e_status = sc1.selectbox(
                                "Status", STATUS_VALUES, format_func=lambda x: x or "— Nessuno —",
                                index=STATUS_VALUES.index(row['status'])
                                if row['status'] in STATUS_VALUES else 0, help=STATUS_HELP)
                            e_watchlist = sc2.checkbox("👁 Watchlist", value=bool(row['watchlist']))
                            e_bottle = sc3.checkbox("⭐ Bottle candidate",
                                                    value=bool(row['bottle_candidate']),
                                                    help="Ignorato se Status = Bottle")
                            col_save, col_cancel = st.columns(2)
                            save_edit = col_save.form_submit_button("💾 Salva", type="primary")
                            cancel_edit = col_cancel.form_submit_button("Annulla")

                            if save_edit:
                                bulk_update({rid: {
                                    'brand': e_brand.strip(), 'name': e_name.strip(),
                                    'form': e_form, 'rating': e_rating, 'comment': e_comment,
                                    'status': e_status, 'watchlist': e_watchlist,
                                    'bottle_candidate': e_bottle and e_status != 'Bottle',
                                }})
                                st.session_state[f"editing_{rid}"] = False
                                st.toast("Modificato!")
                                st.rerun()
                            if cancel_edit:
                                st.session_state[f"editing_{rid}"] = False
                                st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# Tab 2 — Profilo
# ═══════════════════════════════════════════════════════════════════════════════

with tab_profilo:
    st.header("Profilo olfattivo")

    df_tried = tried(df_ratings)

    if df_tried.empty:
        st.info("Nessuna fragranza provata nella lista.")
    else:
        acc = accord_scores(df_tried, ACCORDS)

        # ── Statistiche ───────────────────────────────────────────────────────
        n_pyramid = int((df_tried[['top_notes', 'middle_notes', 'base_notes']]
                         .apply(lambda c: c.str.strip() != '').any(axis=1)).sum())
        fam_pref = (acc.sort_values('liked', ascending=False).iloc[0]['accord'].capitalize()
                    if not acc.empty and acc['liked'].max() > 0 else '—')
        c = st.columns(7)
        c[0].metric("Provati", len(df_tried))
        c[1].metric("Rating medio", f"{df_tried['rating'].mean():.1f}")
        c[2].metric("Bottiglie", int((df_tried['status'] == 'Bottle').sum()))
        c[3].metric("Decant/Sample", int(df_tried['status'].isin(['Decant', 'Sample']).sum()))
        c[4].metric("Bottle candidates", int(df_tried['bottle_candidate'].sum()))
        c[5].metric("Watchlist", int(df_ratings['watchlist'].sum()))
        c[6].metric("Famiglia preferita", fam_pref)
        if n_pyramid < len(df_tried):
            st.caption(f"ℹ️ {len(df_tried) - n_pyramid} profumi hanno solo gli accordi, senza "
                       "piramide: completali in Gestione → Arricchisci note per un profilo più preciso.")

        st.divider()

        # ── Bar chart + Radar ─────────────────────────────────────────────────
        col_left, col_right = st.columns([1, 1])

        with col_left:
            st.subheader("Distribuzione rating")
            counts = df_tried['rating'].value_counts().sort_index()
            fig_bar = go.Figure(go.Bar(
                x=[f"{v:g}" for v in counts.index], y=counts.values,
                marker_color=['#d9534f' if v <= 6 else '#f0ad4e' if v < 7.5 else '#2e9e6b'
                              for v in counts.index],
                hovertemplate='Rating %{x}: %{y} profumi<extra></extra>',
            ))
            fig_bar.update_layout(height=380, margin=dict(l=0, r=0, t=10, b=0),
                                  xaxis_title="Rating", yaxis_title="Profumi",
                                  xaxis_type='category')
            st.plotly_chart(fig_bar, width='stretch', key="rating_dist")
            st.caption("Rosso: ≤ 6 (non piaciuti) · arancio: 6.5–7 · verde: ≥ 7.5 (piaciuti)")

        with col_right:
            st.subheader("Radar olfattivo")
            if acc.empty or acc['all'].sum() == 0:
                st.info("Servono profumi con note per vedere il radar.")
            else:
                top = acc.assign(tot=acc['liked'] + acc['disliked']) \
                         .sort_values('tot', ascending=False).head(10)
                labels = top['accord'].tolist() + [top['accord'].iloc[0]]
                fig_radar = go.Figure()
                for col, name, color, fill in [
                    ('liked', 'Piaciuti (≥ 7.5)', '#2e9e6b', 'rgba(46,158,107,0.20)'),
                    ('disliked', 'Non piaciuti (≤ 6)', '#d9534f', 'rgba(217,83,79,0.15)'),
                ]:
                    vals = top[col].tolist() + [top[col].iloc[0]]
                    fig_radar.add_trace(go.Scatterpolar(
                        r=vals, theta=labels, fill='toself', name=name,
                        line=dict(color=color, width=2), fillcolor=fill,
                        customdata=(top['n_liked' if col == 'liked' else 'n_disliked']
                                    .tolist() + [None]),
                        hovertemplate='<b>%{theta}</b><br>%{customdata} profumi<extra>'
                                      + name + '</extra>',
                    ))
                fig_radar.update_layout(
                    polar=dict(bgcolor='rgba(0,0,0,0)',
                               radialaxis=dict(visible=True, range=[0, 1], showticklabels=False)),
                    height=380, margin=dict(l=40, r=40, t=30, b=30),
                    legend=dict(orientation='h', y=-0.1),
                )
                st.plotly_chart(fig_radar, width='stretch', key="radar_profilo")

        st.divider()

        # ── Scatter ────────────────────────────────────────────────────────────
        st.subheader("La tua collezione nello spazio olfattivo")
        sc1, sc2 = st.columns([3, 1])
        with sc1:
            sf = st.selectbox("Mostra", list(SCATTER_OPTS.keys()),
                              format_func=lambda x: SCATTER_OPTS[x], key='scatter_filter')
        with sc2:
            show_labels = st.toggle("Nomi sul grafico", key="scatter_labels")

        stt = df_tried['status']
        df_for_scatter = {
            'tried_or_owned': df_tried[stt.isin(['Bottle', 'Decant', 'Sample', 'Tested'])],
            'owned':          df_tried[stt.isin(['Bottle', 'Decant', 'Sample'])],
            'bottle_only':    df_tried[stt == 'Bottle'],
        }.get(sf, df_tried)
        df_scatter = compute_scatter_data(df_for_scatter)

        if df_scatter.empty:
            st.caption(f"Nessun profumo con note per il filtro: {SCATTER_OPTS[sf]}.")
        else:
            rng = np.random.default_rng(seed=42)
            x = df_scatter['freshness'] + rng.uniform(-0.04, 0.04, len(df_scatter))
            y = df_scatter['depth'] + rng.uniform(-0.04, 0.04, len(df_scatter))
            symbols = df_scatter['status'].map(
                {'Bottle': 'star', 'Decant': 'circle', 'Sample': 'diamond',
                 'Tested': 'circle-open'}).fillna('circle')
            rmin, rmax = df_scatter['rating'].min(), df_scatter['rating'].max()
            fig_scatter = go.Figure(go.Scatter(
                x=x, y=y,
                mode='markers+text' if show_labels else 'markers',
                text=df_scatter['name'] if show_labels else None,
                textposition='top center', textfont=dict(size=10),
                customdata=np.stack([df_scatter['brand'], df_scatter['name'],
                                     df_scatter['rating'], df_scatter['status'],
                                     df_scatter['id']], axis=-1),
                hovertemplate='<b>%{customdata[0]} — %{customdata[1]}</b><br>'
                              'Rating %{customdata[2]} · %{customdata[3]}<extra></extra>',
                marker=dict(
                    size=df_scatter['intensity'] * 26 + 10, symbol=symbols,
                    color=df_scatter['rating'], colorscale='RdYlGn',
                    cmin=rmin, cmax=rmax if rmax > rmin else rmin + 1,
                    showscale=True, colorbar=dict(title='Rating', thickness=12),
                    line=dict(width=1, color='rgba(0,0,0,0.3)'), opacity=0.85,
                ),
                showlegend=False,
            ))
            fig_scatter.update_layout(
                xaxis=dict(title='Freschezza →', range=[-0.12, 1.12], zeroline=False),
                yaxis=dict(title='Profondità →', range=[-0.12, 1.12], zeroline=False),
                height=560, margin=dict(l=40, r=40, t=20, b=40),
            )
            event = st.plotly_chart(fig_scatter, width='stretch', key="scatter_collezione",
                                    on_select="rerun", selection_mode="points")
            st.caption("Dimensione = intensità · ★ bottiglia · ● decant · ◆ sample · ○ tested. "
                       "Clicca un punto per vederne i dettagli.")
            pts = (event.selection.points if event and event.selection else []) or []
            if pts:
                sel_id = pts[0]['customdata'][4]
                sel = df_ratings[df_ratings['id'] == sel_id]
                if not sel.empty:
                    s = sel.iloc[0]
                    with st.container(border=True):
                        st.markdown(f"**{s['brand']} — {s['name']}** · {fmt_rating(s['rating'])}/10 "
                                    f"{status_icons(s)}")
                        if s['comment']:
                            st.write(s['comment'])
                        show_notes(s['top_notes'], s['middle_notes'], s['base_notes'],
                                   s['main_accords'])

        st.divider()

        # ── Gap analysis ───────────────────────────────────────────────────────
        with st.expander("🔍 Gap analysis — famiglie poco esplorate", expanded=False):
            gaps = gap_analysis(df_tried, df_dataset, ACCORDS)
            gaps = gaps[gaps['mercato'] > 0.02].head(8)
            fig_gap = go.Figure()
            fig_gap.add_trace(go.Bar(y=gaps['accord'], x=gaps['mercato'] * 100,
                                     name='Mercato', orientation='h', marker_color='#9aa5b1'))
            fig_gap.add_trace(go.Bar(y=gaps['accord'], x=gaps['tua_collezione'] * 100,
                                     name='La tua collezione', orientation='h',
                                     marker_color='#2e9e6b'))
            fig_gap.update_layout(barmode='group', height=360, xaxis_title='% di profumi',
                                  yaxis=dict(autorange='reversed'),
                                  margin=dict(l=0, r=0, t=10, b=0),
                                  legend=dict(orientation='h', y=1.08))
            st.plotly_chart(fig_gap, width='stretch', key="gap_chart")
            st.caption("Famiglie molto presenti sul mercato ma rare nella tua lista. "
                       "Puoi esplorarle in Scopri → Esplora uno stile.")

        # ── Deriva temporale ───────────────────────────────────────────────────
        with st.expander("📈 Deriva temporale", expanded=False):
            dated = df_tried[df_tried['date_added'] != ''].copy()
            if len(dated) < 5:
                st.info(f"Da oggi ogni profumo aggiunto riceve la data di inserimento "
                        f"({len(dated)} finora). Il grafico compare da 5 profumi con data.")
            else:
                dated['d'] = pd.to_datetime(dated['date_added'])
                dated = dated.sort_values('d')
                fig_t = go.Figure(go.Scatter(
                    x=dated['d'], y=dated['rating'], mode='markers+lines',
                    text=dated['brand'] + ' ' + dated['name'],
                    hovertemplate='%{text}<br>%{y}<extra></extra>'))
                fig_t.update_layout(height=300, yaxis_title='Rating',
                                    margin=dict(l=0, r=0, t=10, b=0))
                st.plotly_chart(fig_t, width='stretch', key="drift_chart")

        st.divider()

        # ── Analisi LLM ───────────────────────────────────────────────────────
        st.subheader("Analisi del tuo profilo")
        if st.button("🤖 Genera analisi profilo", type="primary", key="btn_profile_analysis"):
            with st.spinner("Analisi in corso…"):
                try:
                    st.session_state['profile_analysis'] = explain_profile(df_tried)
                except Exception as e:
                    st.session_state['profile_analysis'] = f"⚠️ Errore: {e}"
        if 'profile_analysis' in st.session_state:
            st.markdown(st.session_state['profile_analysis'])


# ═══════════════════════════════════════════════════════════════════════════════
# Tab 3 — Scopri
# ═══════════════════════════════════════════════════════════════════════════════

def add_rec_to_watchlist(rec: dict):
    """Bookmarks a recommendation, carrying over its dataset notes."""
    ds_idx = df_dataset.index[(df_dataset['Brand'] == rec['brand']) &
                              (df_dataset['Perfume'] == rec['name'])]
    notes = None
    if len(ds_idx):
        n = dataset_notes(df_dataset.loc[ds_idx[0]])
        notes = {**n, 'ds_brand': rec['brand'], 'ds_name': rec['name'], 'source': 'dataset'}
    add_rating(brand=rec['brand'], name=rec['name'], form='', status='', rating=None,
               watchlist=True, comment='Da Scopri', notes=notes)


def render_rec_list(df_recs: pd.DataFrame, prefix: str, explain: bool):
    known_keys = {(b.lower(), n.lower()) for b, n in zip(df_ratings['brand'], df_ratings['name'])}
    for i, rec in df_recs.iterrows():
        with st.expander(f"**{rec['brand']} — {rec['name']}** · affinità "
                         f"{max(rec['similarity'], 0):.0%}"):
            col1, col2 = st.columns([2, 1])
            with col1:
                show_notes(rec.get('top'), rec.get('middle'), rec.get('base'))
            with col2:
                accs = ', '.join(str(a) for a in (rec.get('accord1'), rec.get('accord2'))
                                 if pd.notna(a) and str(a))
                if accs:
                    st.caption(f"**Accordi:** {accs}")
                st.caption(f"Genere: {rec.get('gender', '—')}")
                if (rec['brand'].lower(), rec['name'].lower()) in known_keys:
                    st.caption("👁 Già in lista")
                elif st.button("👁 Watchlist", key=f"{prefix}_wl_{i}"):
                    add_rec_to_watchlist(rec.to_dict())
                    st.toast(f"{rec['name']} aggiunto alla watchlist")
                    st.rerun()
            if explain:
                expl_key = f"expl_rec_{prefix}_{rec['brand']}_{rec['name']}"
                if st.button("🤖 Spiega perché", key=f"{prefix}_explain_{i}"):
                    with st.spinner("Analisi in corso…"):
                        try:
                            st.session_state[expl_key] = explain_recommendation(
                                rec.to_dict(), tried(df_ratings))
                        except Exception as e:
                            st.session_state[expl_key] = f"⚠️ Errore LLM: {e}"
                if expl_key in st.session_state:
                    st.markdown(st.session_state[expl_key])


with tab_scopri:
    st.header("Scopri")
    df_tried = tried(df_ratings)

    scopri_mode = st.segmented_control(
        "Modalità", options=['profilo', 'simile', 'stile'],
        format_func=lambda x: {'profilo': '🧬 Dal mio profilo',
                               'simile':  '🔍 Simile a…',
                               'stile':   '🧭 Esplora uno stile'}[x],
        default='profilo', key='scopri_mode',
    ) or 'profilo'

    if not df_tried['matched'].any():
        st.info("Servono profumi con note olfattive: abbinali in Gestione → Abbina note.")
    else:
        col1, col2, col3, col4 = st.columns([3, 3, 3, 2])
        with col1:
            n_recs = st.slider("Numero risultati", 5, 25, 10, key="recs_n")
        with col2:
            gender_label = st.selectbox("Genere", list(GENDER_OPTIONS.keys()), index=1,
                                        key="recs_gender")
            gender_val = GENDER_OPTIONS[gender_label]
        with col3:
            quality_pct = st.slider("Filtro qualità", 10, 100, 40, 10, key="recs_quality",
                                    help="10% = solo profumi molto apprezzati · 100% = tutti")
        with col4:
            st.write("")
            niche = st.toggle("💎 Favorisci la nicchia", key="recs_niche",
                              help="La qualità guarda il voto medio, non il numero di recensioni, "
                                   "e i profumi più famosi vengono penalizzati.")

        ref_id = style_label = None
        intensity = 0.5
        exclude_known = True
        if scopri_mode == 'profilo':
            exclude_known = st.checkbox("Escludi profumi già in lista", value=True,
                                        key="recs_exclude")
        elif scopri_mode == 'simile':
            options = df_tried[df_tried['matched']].sort_values('rating', ascending=False)
            ref_labels = {r['id']: f"{label(r)} ({fmt_rating(r['rating'])})"
                          for _, r in options.iterrows()}
            ref_id = st.selectbox("Profumo di riferimento", list(ref_labels),
                                  format_func=lambda i: ref_labels.get(i, i),
                                  key="recs_ref")
        else:
            c1, c2 = st.columns(2)
            style_label = c1.selectbox("Stile da esplorare", list(EXPLORATION_STYLES.keys()),
                                       key="exp_style")
            intensity = c2.slider("Intensità esplorazione", 0.0, 1.0, 0.5, 0.1,
                                  key="exp_intensity",
                                  help="0 = vicino al tuo profilo, 1 = massimo stile nuovo")

        run = st.button("✨ Cerca", type="primary", key="btn_scopri")
        result_key = f"last_{scopri_mode}"
        if run:
            P = get_prepared_dataset()
            k = known_idx_for(df_ratings)
            with st.spinner("Calcolo in corso…"):
                if scopri_mode == 'profilo':
                    res = get_recommendations(df_tried, P, n=n_recs, exclude_known=exclude_known,
                                              gender_filter=gender_val, quality_pct=quality_pct,
                                              known_idx=k, niche=niche)
                elif scopri_mode == 'simile':
                    res = get_similar_to(ref_id, df_tried, P, n=n_recs, quality_pct=quality_pct,
                                         gender_filter=gender_val, known_idx=k,
                                         niche=niche)
                else:
                    res = get_exploration_recommendations(
                        df_tried, P, explore_style=EXPLORATION_STYLES[style_label],
                        intensity=intensity, n=n_recs, gender_filter=gender_val,
                        quality_pct=quality_pct, known_idx=k, niche=niche)
            st.session_state[result_key] = {'df': res, 'style': style_label}
            st.session_state.pop('exp_explanation', None)
            for key in [k_ for k_ in st.session_state if str(k_).startswith(f'expl_rec_{scopri_mode}')]:
                del st.session_state[key]

        saved = st.session_state.get(result_key)
        if saved is not None:
            df_res = saved['df']
            if df_res.empty:
                st.warning("Nessun risultato: prova ad allargare il filtro qualità o il genere.")
            else:
                if scopri_mode == 'stile':
                    st.caption(f"Risultati per lo stile **{saved['style']}**")
                    if st.button("🤖 Spiega questi suggerimenti", key="btn_exp_explain"):
                        with st.spinner("Generando spiegazione…"):
                            try:
                                st.session_state['exp_explanation'] = explain_exploration(
                                    saved['style'], df_tried, df_res)
                            except Exception as e:
                                st.session_state['exp_explanation'] = f"⚠️ LLM non disponibile: {e}"
                    if 'exp_explanation' in st.session_state:
                        st.info(st.session_state['exp_explanation'])
                render_rec_list(df_res, scopri_mode, explain=scopri_mode != 'stile')


# ═══════════════════════════════════════════════════════════════════════════════
# Tab 4 — Gestione
# ═══════════════════════════════════════════════════════════════════════════════

def render_match_picker(row, key_prefix: str):
    """Candidate list for one entry, with confirm / skip actions."""
    rid = row['id']
    cands = get_candidates_cached(row['brand'], row['name'], 10)
    if not cands:
        st.warning("Nessun candidato trovato nel dataset.")
    for c in cands:
        col1, col2 = st.columns([4, 1])
        with col1:
            current = (row['match_status'] == 'matched' and c['brand'] == row['ds_brand']
                       and c['name'] == row['ds_name'])
            st.markdown(f"**{c['brand']} — {c['name']}** · somiglianza {c['score']}%"
                        + ("  ✅ *attuale*" if current else ""))
            show_notes(c['top'], c['middle'], c['base'], c['accords'])
            if not (c['top'] or c['middle'] or c['base']):
                st.caption("Solo accordi nel dataset: le note esistenti verranno mantenute.")
        with col2:
            if st.button("✓ Questo", key=f"{key_prefix}_{rid}_{c['dataset_idx']}"):
                apply_match(rid, c['brand'], c['name'],
                            {k: c[k] for k in ('top', 'middle', 'base', 'accords')})
                st.toast(f"Abbinato a {c['brand']} — {c['name']}")
                st.session_state.pop('just_added', None)
                st.rerun()
    b1, b2 = st.columns(2)
    if b1.button("⏭️ Non è nel dataset", key=f"{key_prefix}_skip_{rid}"):
        set_match_status(rid, 'skipped')
        st.session_state.pop('just_added', None)
        st.rerun()
    if row['match_status'] == 'matched' and b2.button("✖ Rimuovi abbinamento",
                                                      key=f"{key_prefix}_unmatch_{rid}"):
        set_match_status(rid, '')
        st.rerun()


with tab_gestione:
    st.header("Gestione")

    # ── Duplicate dialog BEFORE inner tabs so buttons survive reruns ──────────
    if 'pending_duplicate' in st.session_state:
        _pd = st.session_state['pending_duplicate']
        st.warning(f"⚠️ **{_pd['brand']} — {_pd['name']} {_pd['form']}** è già nella tua lista. "
                   "Aggiungerlo comunque?")
        _dc1, _dc2 = st.columns(2)
        if _dc1.button("✅ Aggiungi comunque", key="force_add_btn", type="primary"):
            st.session_state['just_added'] = add_rating(**_pd)
            st.session_state.pop('pending_duplicate', None)
            st.rerun()
        if _dc2.button("❌ Annulla", key="cancel_add_btn"):
            st.session_state.pop('pending_duplicate', None)
            st.rerun()

    health = match_health(df_ratings, df_dataset)
    n_serious = int((health['gravità'] != 'bassa').sum()) if not health.empty else 0

    gtab_aggiungi, gtab_match, gtab_arricchisci, gtab_salute = st.tabs([
        "➕ Aggiungi", "🔗 Abbina note", "🔍 Arricchisci note",
        f"🩺 Salute dati{f' ({n_serious})' if n_serious else ''}",
    ])

    # ─── Aggiungi ─────────────────────────────────────────────────────────────
    with gtab_aggiungi:
        with st.form("add_fragrance", clear_on_submit=True):
            col1, col2 = st.columns(2)
            with col1:
                new_brand = st.text_input("Brand *")
                new_name = st.text_input("Nome *")
                new_form = st.selectbox("Formato", FORMS)
            with col2:
                new_status = st.selectbox("Status", STATUS_VALUES,
                                          format_func=lambda x: x or "— Nessuno —",
                                          help=STATUS_HELP)
                wc1, wc2 = st.columns(2)
                new_watchlist = wc1.checkbox("👁 Watchlist", help="Segnalibro, con qualsiasi status")
                new_bottle = wc2.checkbox("⭐ Bottle candidate", help="Ignorato se Status = Bottle")
                not_tried = st.checkbox("Non ancora provato (senza voto)")
                new_rating = st.select_slider("Rating", options=RATING_OPTIONS, value=7.0)
            new_comment = st.text_area("Commento",
                                       placeholder="Impressioni, occasioni d'uso, durata…")
            submitted = st.form_submit_button("➕ Aggiungi", type="primary")

            if submitted:
                data = dict(brand=new_brand, name=new_name, form=new_form, status=new_status,
                            watchlist=new_watchlist or not_tried,
                            rating=None if not_tried else new_rating,
                            comment=new_comment, bottle_candidate=new_bottle)
                if not new_brand.strip() or not new_name.strip():
                    st.error("Brand e nome sono obbligatori.")
                elif check_duplicate(new_brand, new_name, new_form):
                    st.session_state['pending_duplicate'] = data
                    st.rerun()
                else:
                    st.session_state['just_added'] = add_rating(**data)
                    st.rerun()

        just = st.session_state.get('just_added')
        if just and (df_ratings['id'] == just).any():
            row = df_ratings[df_ratings['id'] == just].iloc[0]
            st.success(f"✓ {row['brand']} {row['name']} aggiunto. "
                       "Abbinalo subito al dataset per ottenere le note:")
            with st.container(border=True):
                render_match_picker(row, "quick")
            if st.button("Più tardi", key="quick_later"):
                st.session_state.pop('just_added', None)
                st.rerun()

    # ─── Abbina note ──────────────────────────────────────────────────────────
    with gtab_match:
        st.caption("Collega ogni profumo al dataset per ottenere piramide e accordi.")
        ms = df_ratings['match_status']
        c = st.columns(3)
        c[0].metric("Abbinati", int((ms == 'matched').sum()))
        c[1].metric("Non nel dataset", int((ms == 'skipped').sum()))
        c[2].metric("Da fare", int((ms == '').sum()))

        only_pending = st.toggle("Solo da abbinare", value=bool((ms == '').any()),
                                 key="match_only_pending")
        pool = df_ratings[ms == ''] if only_pending else df_ratings
        if pool.empty:
            st.success("Niente da abbinare 🎉")
        else:
            pool = pool.sort_values(['brand', 'name'])
            icon = {'matched': '✅', 'skipped': '⏭️', '': '⏳'}
            match_labels = {r['id']: f"{icon[r['match_status']]} {label(r)}"
                            for _, r in pool.iterrows()}
            sel_id = st.selectbox("Profumo", list(match_labels),
                                  format_func=lambda i: match_labels.get(i, i),
                                  key="match_target")
            row = df_ratings[df_ratings['id'] == sel_id].iloc[0]
            st.caption(f"Il tuo voto: {fmt_rating(row['rating'])} · {notes_provenance(row)}")
            render_match_picker(row, "match")

        with st.expander("🔍 Cerca manualmente nel dataset"):
            search_query = st.text_input("Cerca per nome o brand",
                                         placeholder="es. philosykos, diptyque, fig…")
            if search_query:
                q = search_query.lower()
                df_search = df_dataset[df_dataset['name_clean'].str.contains(q, na=False, regex=False) |
                                       df_dataset['brand_clean'].str.contains(q, na=False, regex=False)
                                       ].head(50)
                if df_search.empty:
                    st.caption("Nessun risultato.")
                else:
                    all_labels = {r['id']: label(r) for _, r in df_ratings.iterrows()}
                    target = st.selectbox("Abbina a quale tuo profumo?", list(all_labels),
                                          format_func=lambda i: all_labels.get(i, i),
                                          key="manual_target")
                    for ds_idx, sr in df_search.iterrows():
                        col1, col2 = st.columns([4, 1])
                        n = dataset_notes(sr)
                        col1.caption(f"**{sr['Brand']} — {sr['Perfume']}** | Top: {n['top'] or '—'} | "
                                     f"Cuore: {n['middle'] or '—'} | Fondo: {n['base'] or '—'} | "
                                     f"{n['accords']}")
                        if col2.button("✓ Usa", key=f"manual_{ds_idx}"):
                            apply_match(target, sr['Brand'], sr['Perfume'], n)
                            st.toast("Abbinato!")
                            st.rerun()

    # ─── Arricchisci note ─────────────────────────────────────────────────────
    with gtab_arricchisci:
        st.caption("Completa le note dei profumi senza piramide: cerca su Fragrantica "
                   "o incollale a mano.")
        no_pyr = (df_ratings[['top_notes', 'middle_notes', 'base_notes']]
                  .apply(lambda c: c.str.strip() == '').all(axis=1))
        show_all = st.toggle("Mostra tutti i profumi", value=not no_pyr.any(), key="enrich_all")
        pool = df_ratings if show_all else df_ratings[no_pyr]
        st.caption(f"{int(no_pyr.sum())} profumi senza piramide.")

        if not pool.empty:
            pool = pool.sort_values(['brand', 'name'])
            enrich_labels = {r['id']: label(r) for _, r in pool.iterrows()}
            eid = st.selectbox("Profumo", list(enrich_labels),
                               format_func=lambda i: enrich_labels.get(i, i),
                               key="enrich_target")
            row = df_ratings[df_ratings['id'] == eid].iloc[0]
            st.caption(notes_provenance(row))
            result_key = f"enrich_result_{eid}"

            if st.button("🌐 Cerca su Fragrantica", key=f"search_{eid}"):
                with st.spinner("Cerco su Fragrantica…"):
                    result = enrich_from_web(row['brand'], row['name'])
                st.session_state[result_key] = result if result else 'not_found'

            res = st.session_state.get(result_key)
            if res == 'not_found':
                st.warning("Nessun risultato. Fragrantica spesso blocca le richieste automatiche "
                           "(errore 403): apri la pagina nel browser e incolla le note qui sotto.")
            elif isinstance(res, dict):
                url_input = st.text_input("URL Fragrantica", value=res['url'], key=f"url_{eid}")
                if url_input != res['url'] and st.button("🔄 Ricarica", key=f"reload_{eid}"):
                    with st.spinner("Carico…"):
                        new_res = scrape_fragrantica(url_input)
                    if new_res:
                        st.session_state[result_key] = {**new_res, 'url': url_input}
                        st.rerun()
                    else:
                        st.error("Nessuna nota trovata a quell'URL.")
                show_notes(res['top'], res['middle'], res['base'], res['accords'])
                if st.button("✅ Salva queste note", type="primary", key=f"save_web_{eid}"):
                    save_enriched_notes(eid, res['top'], res['middle'], res['base'],
                                        res['accords'], url=res['url'])
                    st.session_state.pop(result_key, None)
                    st.toast("Note salvate!")
                    st.rerun()

            with st.form(f"manual_notes_{eid}"):
                st.markdown("**Inserisci o correggi a mano**")
                m_top = st.text_input("Top", value=row['top_notes'])
                m_mid = st.text_input("Cuore", value=row['middle_notes'])
                m_base = st.text_input("Fondo", value=row['base_notes'])
                m_acc = st.text_input("Accordi", value=row['main_accords'],
                                      help="Separati da virgola, es. woody, warm spicy, leather")
                m_url = st.text_input("URL fonte (opzionale)", value=row['notes_url'])
                if st.form_submit_button("💾 Salva note", type="primary"):
                    set_notes(eid, m_top, m_mid, m_base, m_acc,
                              source='web_enriched' if m_url else 'manual', url=m_url)
                    st.toast("Note salvate!")
                    st.rerun()

    # ─── Salute dati ──────────────────────────────────────────────────────────
    with gtab_salute:
        st.caption("Controlli automatici sui tuoi dati: abbinamenti sospetti, "
                   "note che non tornano, profumi senza note.")
        if health.empty:
            st.success("Tutto in ordine ✅")
        else:
            order = {'alta': 0, 'media': 1, 'bassa': 2}
            st.dataframe(health.sort_values('gravità', key=lambda s: s.map(order))
                         .drop(columns=['id']),
                         hide_index=True, width='stretch')
            st.caption("Correggi da Abbina note (abbinamenti) o Arricchisci note (note mancanti).")

        st.divider()
        st.markdown("**Backup**")
        with open(MY_RATINGS_FILE, 'rb') as f:
            st.download_button("⬇️ Scarica my_ratings.csv", f.read(),
                               file_name=f"my_ratings_{date.today().isoformat()}.csv",
                               mime='text/csv')
