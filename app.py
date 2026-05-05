import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from config import ACCORDS, EXPLORATION_STYLES, GENDER_OPTIONS, MIN_RATING, MAX_RATING, RATING_STEP
from enricher import enrich_from_web, scrape_fragrantica
from ratings import load_ratings, save_ratings, add_rating, save_enriched_notes, delete_rating, check_duplicate
from matcher import load_dataset, get_candidates, enrich_ratings, save_confirmed_matches, load_confirmed_matches
from recommender import get_recommendations, get_similar_to, get_exploration_recommendations, build_personal_profile, compute_scatter_data
from llm import explain_recommendation, explain_profile, explain_exploration

# ── Setup ─────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Fragrance Explorer",
    page_icon="🌿",
    layout="wide"
)

@st.cache_data
def load_dataset_cached():
    return load_dataset()

df_dataset = load_dataset_cached()

_SCATTER_OPTS = {
    'all':            'Tutti',
    'tried_or_owned': 'Provati o posseduti (Bottle+Decant+Sample+Tested)',
    'owned':          'Solo posseduti (Bottle+Decant+Sample)',
    'bottle_only':    'Solo bottiglie',
}
_STATUS_ICONS = {'Bottle': '🫙', 'Decant': '🧪', 'Sample': '🧴', 'Tested': '✓'}

# ── Tab structure ─────────────────────────────────────────────────────────────

tab_lista, tab_profilo, tab_scopri, tab_gestione = st.tabs([
    "📋 La mia lista",
    "🧬 Profilo",
    "✨ Scopri",
    "⚙️ Gestione",
])

# ═══════════════════════════════════════════════════════════════════════════════
# Tab 1 — La mia lista
# ═══════════════════════════════════════════════════════════════════════════════

with tab_lista:
    st.header("La mia lista")

    df_ratings = load_ratings()

    if df_ratings.empty:
        st.info("Nessuna fragranza ancora. Usa il tab Gestione per aggiungerne.")
    else:
        # ── Visibilità grafico (session_state condiviso con Profilo) ──────────
        st.selectbox(
            "Visibilità grafico (usata anche nel Profilo)",
            options=list(_SCATTER_OPTS.keys()),
            format_func=lambda x: _SCATTER_OPTS[x],
            key='scatter_filter',
        )

        st.divider()

        # ── Filtri lista ──────────────────────────────────────────────────────
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            rating_min = st.slider("Rating minimo", MIN_RATING, MAX_RATING,
                                   MIN_RATING, RATING_STEP)
        with col2:
            _STATUS_OPTS = ["Tutti", "Bottle", "Decant", "Sample", "Tested",
                            "Watchlist", "Nessuno status"]
            status_filter = st.selectbox("Status", _STATUS_OPTS)
        with col3:
            only_bc = st.checkbox("Solo bottle candidates")
        with col4:
            sort_by = st.selectbox("Ordina per", ["Rating ↓", "Rating ↑", "Brand A-Z"])

        df_view = df_ratings.copy()
        _st_col = df_view.get('status', pd.Series('', index=df_view.index)).fillna('')
        if status_filter == "Watchlist":
            df_view = df_view[
                df_view.get('watchlist', pd.Series(False, index=df_view.index))
                .fillna(False).astype(bool)
            ]
        elif status_filter == "Nessuno status":
            df_view = df_view[_st_col.astype(str).str.strip() == '']
        elif status_filter != "Tutti":
            df_view = df_view[_st_col.astype(str) == status_filter]
        if only_bc:
            df_view = df_view[df_view['bottle_candidate'] == True]
        df_view = df_view[df_view['rating'] >= rating_min]

        if sort_by == "Rating ↓":
            df_view = df_view.sort_values('rating', ascending=False)
        elif sort_by == "Rating ↑":
            df_view = df_view.sort_values('rating', ascending=True)
        else:
            df_view = df_view.sort_values('brand')

        st.caption(f"{len(df_view)} fragranze")

        # ── Lista ─────────────────────────────────────────────────────────────
        for idx, row in df_view.iterrows():
            _s = str(row.get('status', '') or '')
            _icons = (
                _STATUS_ICONS.get(_s, '')
                + ('⭐' if row.get('bottle_candidate') and _s != 'Bottle' else '')
                + ('👁' if row.get('watchlist') else '')
            )
            with st.expander(
                f"**{row['rating']}** — {row['brand']} {row['name']}"
                + (f" {_icons}" if _icons else "")
            ):
                col1, col2 = st.columns([2, 1])
                with col1:
                    if row.get('comment'):
                        st.write(row['comment'])
                    if row.get('top_notes'):
                        st.caption(f"Top: {row['top_notes']}")
                    if row.get('middle_notes'):
                        st.caption(f"Middle: {row['middle_notes']}")
                    if row.get('base_notes'):
                        st.caption(f"Base: {row['base_notes']}")
                with col2:
                    st.caption(f"Forma: {row.get('form', '—')}")
                    _s = str(row.get('status', '') or '')
                    st.caption(f"Status: {_s or '—'}"
                               + (" · Watchlist" if row.get('watchlist') else ""))
                    if row.get('matched', False):
                        st.caption("✅ Note abbinate")
                    else:
                        st.caption("⚠️ Note non abbinate")

                    col_edit, col_del = st.columns(2)
                    with col_edit:
                        if st.button("✏️ Modifica", key=f"edit_{idx}"):
                            st.session_state[f"editing_{idx}"] = True
                    with col_del:
                        if not st.session_state.get(f"confirm_delete_{idx}"):
                            if st.button("🗑️", key=f"del_{idx}"):
                                st.session_state[f"confirm_delete_{idx}"] = True
                                st.rerun()
                        else:
                            st.warning("Sicuro?")
                            col_yes, col_no = st.columns(2)
                            with col_yes:
                                if st.button("✅", key=f"del_confirm_{idx}",
                                             type="primary"):
                                    delete_rating(idx)
                                    st.session_state.pop(f"confirm_delete_{idx}", None)
                                    st.rerun()
                            with col_no:
                                if st.button("❌", key=f"del_cancel_{idx}"):
                                    st.session_state.pop(f"confirm_delete_{idx}", None)
                                    st.rerun()

                if st.session_state.get(f"editing_{idx}"):
                    with st.form(key=f"form_edit_{idx}"):
                        e_rating = st.select_slider(
                            "Rating",
                            options=[r/10 for r in range(
                                int(MIN_RATING*10),
                                int(MAX_RATING*10)+1,
                                int(RATING_STEP*10)
                            )],
                            value=float(row['rating'])
                        )
                        e_comment = st.text_area("Commento",
                                                  value=row.get('comment', ''))
                        _STATUS_LIST = ["", "Bottle", "Decant", "Sample", "Tested"]
                        _cur_status = str(row.get('status', '') or '')
                        e_status = st.selectbox(
                            "Status",
                            _STATUS_LIST,
                            format_func=lambda x: x or "— Nessuno —",
                            index=_STATUS_LIST.index(_cur_status)
                            if _cur_status in _STATUS_LIST else 0
                        )
                        e_watchlist = st.checkbox(
                            "Watchlist",
                            value=bool(row.get('watchlist', False))
                        )
                        e_bottle = st.checkbox(
                            "Bottle candidate (N/A se Status = Bottle)",
                            value=bool(row.get('bottle_candidate'))
                        )
                        col_save, col_cancel = st.columns(2)
                        with col_save:
                            save_edit = st.form_submit_button("💾 Salva",
                                                               type="primary")
                        with col_cancel:
                            cancel_edit = st.form_submit_button("Annulla")

                        if save_edit:
                            df_ratings.at[idx, 'rating'] = e_rating
                            df_ratings.at[idx, 'comment'] = e_comment
                            df_ratings.at[idx, 'status'] = e_status
                            df_ratings.at[idx, 'watchlist'] = e_watchlist
                            df_ratings.at[idx, 'bottle_candidate'] = e_bottle
                            save_ratings(df_ratings)
                            st.session_state[f"editing_{idx}"] = False
                            st.success("Modificato!")
                            st.rerun()

                        if cancel_edit:
                            st.session_state[f"editing_{idx}"] = False
                            st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# Tab 2 — Profilo
# ═══════════════════════════════════════════════════════════════════════════════

with tab_profilo:
    st.header("Profilo olfattivo")

    df_ratings = load_ratings()

    if df_ratings.empty:
        st.info("Nessuna fragranza nella lista.")
    else:
        # ── Pre-calcolo top_accords ───────────────────────────────────────────
        df_matched = df_ratings[df_ratings['matched'] == True] \
            if 'matched' in df_ratings.columns else pd.DataFrame()

        top_accords = []
        if not df_matched.empty:
            accord_scores = {a: 0.0 for a in ACCORDS}
            for _, row in df_matched.iterrows():
                note_str = ' '.join(filter(None, [
                    str(row.get('top_notes', '')),
                    str(row.get('middle_notes', '')),
                    str(row.get('base_notes', '')),
                    str(row.get('main_accords', '')),
                ])).lower()
                weight = (row['rating'] - 1) / 9
                for accord in ACCORDS:
                    if accord in note_str:
                        accord_scores[accord] += weight
            max_score = max(accord_scores.values()) or 1
            norm_scores = {a: v / max_score for a, v in accord_scores.items()}
            top_accords = sorted(norm_scores.items(),
                                 key=lambda x: x[1], reverse=True)[:8]

        # ── Statistiche ───────────────────────────────────────────────────────
        st.subheader("Statistiche")

        _status_s         = df_ratings.get('status', pd.Series('', index=df_ratings.index)).fillna('')
        total             = len(df_ratings)
        avg_rating        = df_ratings['rating'].mean()
        bottles           = int((_status_s == 'Bottle').sum())
        decants           = int((_status_s == 'Decant').sum())
        bottle_candidates = int((df_ratings['bottle_candidate'] == True).sum())
        fam_preferita     = top_accords[0][0].capitalize() if top_accords else '—'

        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Totale",             total)
        c2.metric("Avg rating",         f"{avg_rating:.1f}")
        c3.metric("Bottiglie",          bottles)
        c4.metric("Decants",            decants)
        c5.metric("Bottle Candidates",  bottle_candidates)
        c6.metric("Famiglia preferita", fam_preferita)

        st.divider()

        # ── Bar chart + Radar ─────────────────────────────────────────────────
        col_left, col_right = st.columns([1, 1])

        with col_left:
            st.subheader("Distribuzione rating")
            rating_counts = df_ratings['rating'].value_counts().sort_index(ascending=False)
            fig_bar = go.Figure(go.Bar(
                x=rating_counts.values,
                y=[str(r) for r in rating_counts.index],
                orientation='h',
                marker_color='#5090ff',
                hovertemplate='Rating %{y} — %{x} profumi<extra></extra>',
            ))
            fig_bar.update_layout(
                height=350,
                margin=dict(l=0, r=0, t=10, b=0),
                xaxis_title="Numero profumi",
                yaxis_title="Rating",
            )
            st.plotly_chart(fig_bar, width='stretch', key="rating_dist")

        with col_right:
            st.subheader("Radar olfattivo")
            if df_matched.empty:
                st.info("Abbina le note in Gestione → Abbina note per vedere il radar.")
            else:
                labels = [a for a, _ in top_accords]
                values = [v for _, v in top_accords]
                labels_closed = labels + [labels[0]]
                values_closed = values + [values[0]]

                fig_radar = go.Figure()
                fig_radar.add_trace(go.Scatterpolar(
                    r=values_closed,
                    theta=labels_closed,
                    fill='toself',
                    line=dict(color='#00c878', width=2),
                    fillcolor='rgba(0,200,120,0.18)',
                    name='Il tuo profilo',
                    hovertemplate='<b>Famiglia: %{theta}</b><br>Affinità: %{r:.0%}<extra></extra>',
                ))
                fig_radar.update_layout(
                    polar=dict(
                        bgcolor='rgba(240,240,240,0.1)',
                        radialaxis=dict(
                            visible=True,
                            range=[0, 1],
                            tickfont=dict(size=10, color='#666666'),
                        ),
                        angularaxis=dict(tickfont=dict(size=12))
                    ),
                    height=350,
                    margin=dict(l=40, r=40, t=40, b=40),
                    showlegend=False,
                )
                st.plotly_chart(fig_radar, width='stretch', key="radar_profilo")

        st.divider()

        # ── Scatter — usa il filtro da session_state (impostato in La mia lista)
        st.subheader("La tua collezione nello spazio olfattivo")

        _sf = st.session_state.get('scatter_filter', 'all')
        _sf_status = df_ratings.get('status', pd.Series('', index=df_ratings.index)).fillna('')
        if _sf == 'tried_or_owned':
            df_for_scatter = df_ratings[_sf_status.isin(['Bottle', 'Decant', 'Sample', 'Tested'])]
        elif _sf == 'owned':
            df_for_scatter = df_ratings[_sf_status.isin(['Bottle', 'Decant', 'Sample'])]
        elif _sf == 'bottle_only':
            df_for_scatter = df_ratings[_sf_status == 'Bottle']
        else:
            df_for_scatter = df_ratings

        df_scatter = compute_scatter_data(df_for_scatter)

        if df_scatter.empty:
            st.caption(f"Abbina qualche profumo per vedere il grafico "
                       f"(filtro attuale: {_SCATTER_OPTS.get(_sf, _sf)}).")
        else:
            rng = np.random.default_rng(seed=42)
            jitter = 0.05
            x = df_scatter['freshness'] + rng.uniform(-jitter, jitter, len(df_scatter))
            y = df_scatter['depth']     + rng.uniform(-jitter, jitter, len(df_scatter))

            fig_scatter = go.Figure()
            fig_scatter.add_trace(go.Scatter(
                x=x, y=y,
                mode='markers',
                name='Profumi',
                showlegend=False,
                marker=dict(
                    size=df_scatter['intensity'] * 28 + 10,
                    color=df_scatter['rating'],
                    colorscale='RdYlGn',
                    cmin=1, cmax=10,
                    showscale=True,
                    colorbar=dict(title='Rating', thickness=12),
                    line=dict(width=1, color='rgba(0,0,0,0.2)'),
                    opacity=0.85,
                ),
                text=df_scatter.apply(
                    lambda r: f"{r['brand']} — {r['name']}<br>Rating: {r['rating']}", axis=1
                ),
                hovertemplate='%{text}<extra></extra>',
            ))
            for label, size in [(' ', 6), (' ', 12), ('Intensità: bassa / media / alta', 18)]:
                fig_scatter.add_trace(go.Scatter(
                    x=[None], y=[None],
                    mode='markers',
                    name=label,
                    marker=dict(size=size, color='rgba(120,120,120,0.6)',
                                line=dict(width=1, color='rgba(0,0,0,0.2)')),
                ))
            fig_scatter.update_layout(
                xaxis=dict(title='Freschezza', range=[-0.15, 1.15], zeroline=False),
                yaxis=dict(title='Profondità', range=[-0.15, 1.15], zeroline=False),
                legend=dict(orientation='h', x=0, y=1.08,
                            xanchor='left', yanchor='bottom'),
                height=500,
                margin=dict(l=40, r=40, t=60, b=40),
                plot_bgcolor='rgba(240,240,240,0.1)',
            )
            st.plotly_chart(fig_scatter, width='stretch', key="scatter_collezione")

        st.divider()

        # ── Stubs sezioni future ──────────────────────────────────────────────
        with st.expander("📈 Deriva temporale [coming soon]"):
            st.info("In arrivo: analisi dell'evoluzione dei tuoi gusti nel tempo.")

        with st.expander("🔍 Gap analysis [coming soon]"):
            st.info("In arrivo: famiglie olfattive non ancora esplorate nella tua collezione.")

        st.divider()

        # ── Analisi LLM ───────────────────────────────────────────────────────
        st.subheader("Analisi del tuo profilo")
        if st.button("🤖 Genera analisi profilo", type="primary", key="btn_profile_analysis"):
            with st.spinner("Analisi in corso…"):
                try:
                    analysis = explain_profile(df_ratings)
                    st.session_state['profile_analysis'] = analysis
                except Exception as e:
                    st.session_state['profile_analysis'] = f"⚠️ Errore: {e}"
        if 'profile_analysis' in st.session_state:
            st.markdown(st.session_state['profile_analysis'])


# ═══════════════════════════════════════════════════════════════════════════════
# Tab 3 — Scopri (Raccomandazioni + Esplora stili)
# ═══════════════════════════════════════════════════════════════════════════════

with tab_scopri:
    st.header("Scopri")

    df_ratings = load_ratings()

    scopri_mode = st.radio(
        "Modalità",
        options=['profilo', 'stile'],
        format_func=lambda x: {
            'profilo': '🧬 Basato sul tuo profilo',
            'stile':   '🧭 Esplora uno stile',
        }[x],
        horizontal=True,
        key='scopri_mode',
    )

    st.divider()

    # ─── Modalità: Raccomandazioni ────────────────────────────────────────────
    if scopri_mode == 'profilo':

        if 'matched' not in df_ratings.columns or df_ratings['matched'].sum() == 0:
            st.info("Abbina almeno qualche profumo in Gestione → Abbina note per ottenere raccomandazioni.")
        else:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                n_recs = st.slider("Numero raccomandazioni", 5, 20, 10, key="recs_n")
            with col2:
                gender_label = st.selectbox("Genere", list(GENDER_OPTIONS.keys()),
                                             key="recs_gender")
                gender_val = GENDER_OPTIONS[gender_label]
            with col3:
                exclude_known = st.checkbox("Escludi profumi già noti", value=True,
                                             key="recs_exclude")
            with col4:
                quality_pct = st.slider(
                    "Filtro qualità", 10, 100, 40, 10,
                    key="recs_quality",
                    help="10% = solo profumi top | 40% = equilibrato | 100% = tutti"
                )

            st.divider()

            recs_mode = st.radio(
                "Tipo di raccomandazione",
                options=['profilo', 'simile'],
                format_func=lambda x: {
                    'profilo': '🧬 Basato sul mio profilo',
                    'simile':  '🔍 Simile a un profumo specifico',
                }[x],
                horizontal=True,
                key='recs_mode',
            )

            ref_brand = ref_name = None
            if recs_mode == 'simile':
                known = df_ratings[['brand', 'name']].apply(
                    lambda r: f"{r['brand']} — {r['name']}", axis=1
                ).tolist()
                selected = st.selectbox("Scegli profumo di riferimento", known,
                                         key="recs_ref")
                ref_brand, ref_name = selected.split(' — ', 1)

            if st.button("✨ Genera raccomandazioni", type="primary",
                         key="btn_generate_recs"):
                with st.spinner("Calcolo in corso…"):
                    if recs_mode == 'profilo':
                        _df_recs = get_recommendations(
                            df_ratings, df_dataset, n=n_recs,
                            exclude_known=exclude_known,
                            gender_filter=gender_val,
                            quality_pct=quality_pct
                        )
                    else:
                        _df_recs = get_similar_to(
                            ref_brand, ref_name,
                            df_ratings, df_dataset, n=n_recs,
                            quality_pct=quality_pct
                        )
                st.session_state['last_recs'] = _df_recs
                for k in list(st.session_state.keys()):
                    if k.startswith('expl_rec_'):
                        del st.session_state[k]

            df_recs = st.session_state.get('last_recs', pd.DataFrame())

            if not df_recs.empty:
                for _i, rec in df_recs.iterrows():
                    with st.expander(
                        f"**{rec['brand']} — {rec['name']}** "
                        f"(similarità: {rec['similarity']:.2f})"
                    ):
                        col1, col2 = st.columns([2, 1])
                        with col1:
                            if pd.notna(rec.get('top')):
                                st.caption(f"Top: {rec['top']}")
                            if pd.notna(rec.get('middle')):
                                st.caption(f"Middle: {rec['middle']}")
                            if pd.notna(rec.get('base')):
                                st.caption(f"Base: {rec['base']}")
                        with col2:
                            if pd.notna(rec.get('accord1')):
                                st.caption(f"Accord: {rec['accord1']}")
                            if pd.notna(rec.get('accord2')):
                                st.caption(f"{rec['accord2']}")
                            st.caption(f"Genere: {rec.get('gender', '—')}")

                        _expl_key = f"expl_rec_{_i}_{rec['brand']}_{rec['name']}"
                        if st.button("🤖 Spiega perché",
                                     key=f"explain_{_i}_{rec['brand']}_{rec['name']}"):
                            with st.spinner("Analisi in corso…"):
                                try:
                                    _expl = explain_recommendation(rec.to_dict(), df_ratings)
                                    st.session_state[_expl_key] = _expl
                                except Exception as e:
                                    st.session_state[_expl_key] = f"⚠️ Errore LLM: {e}"
                            st.rerun()
                        if _expl_key in st.session_state:
                            st.markdown(st.session_state[_expl_key])
            elif 'last_recs' in st.session_state:
                st.warning("Nessuna raccomandazione trovata.")

    # ─── Modalità: Esplora stili ──────────────────────────────────────────────
    else:
        st.caption("Scopri profumi in territori olfattivi nuovi, ancorati al tuo gusto.")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            style_label = st.selectbox("Stile da esplorare",
                                       list(EXPLORATION_STYLES.keys()),
                                       key="exp_style")
        with col2:
            intensity = st.slider(
                "Intensità esplorazione", 0.0, 1.0, 0.5, 0.1,
                key="exp_intensity",
                help="0 = vicino al tuo profilo, 1 = massimo stile nuovo"
            )
        with col3:
            gender_label_e = st.selectbox("Genere", list(GENDER_OPTIONS.keys()),
                                          key="gender_explore")
            gender_val_e = GENDER_OPTIONS[gender_label_e]
        with col4:
            quality_pct_e = st.slider(
                "Filtro qualità %", 10, 100, 40, 10,
                key="quality_explore",
                help="10% = solo profumi top | 40% = equilibrato | 100% = tutti"
            )

        st.divider()

        if st.button("🧭 Esplora", type="primary", key="btn_esplora"):
            with st.spinner("Calcolo in corso…"):
                df_exp = get_exploration_recommendations(
                    df_ratings, df_dataset,
                    explore_style=EXPLORATION_STYLES[style_label],
                    intensity=intensity,
                    n=10,
                    gender_filter=gender_val_e,
                    quality_pct=quality_pct_e
                )
            st.session_state['last_exp'] = df_exp

        df_exp = st.session_state.get('last_exp', pd.DataFrame())

        if not df_exp.empty:
            if 'matched' in df_ratings.columns and df_ratings['matched'].sum() > 0:
                with st.spinner("Generando spiegazione…"):
                    try:
                        explanation = explain_exploration(style_label, df_ratings, df_exp)
                        st.info(explanation)
                    except Exception as e:
                        st.caption(f"LLM non disponibile: {e}")

            st.divider()

            for _, rec in df_exp.iterrows():
                with st.expander(
                    f"**{rec['brand']} — {rec['name']}** "
                    f"(similarità: {rec['similarity']:.2f})"
                ):
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        if pd.notna(rec.get('top')):
                            st.caption(f"Top: {rec['top']}")
                        if pd.notna(rec.get('middle')):
                            st.caption(f"Middle: {rec['middle']}")
                        if pd.notna(rec.get('base')):
                            st.caption(f"Base: {rec['base']}")
                    with col2:
                        if pd.notna(rec.get('accord1')):
                            st.caption(f"Accord: {rec['accord1']}")
                        st.caption(f"Genere: {rec.get('gender', '—')}")
        elif 'last_exp' in st.session_state:
            st.warning("Nessun risultato trovato.")


# ═══════════════════════════════════════════════════════════════════════════════
# Tab 4 — Gestione (Aggiungi + Abbina note + Arricchisci note)
# ═══════════════════════════════════════════════════════════════════════════════

with tab_gestione:
    st.header("Gestione")

    # ── Duplicate dialog BEFORE inner tabs so buttons survive reruns ──────────
    if 'pending_duplicate' in st.session_state:
        _pd = st.session_state['pending_duplicate']
        st.warning(
            f"⚠️ **{_pd['brand']} — {_pd['name']}** è già nella tua lista. "
            "Vuoi aggiungerlo comunque (es. versione diversa o formato diverso)?"
        )
        _dc1, _dc2 = st.columns(2)
        with _dc1:
            if st.button("✅ Aggiungi comunque", key="force_add_btn", type="primary"):
                add_rating(**{k: v for k, v in _pd.items() if k != '_label'})
                st.success(f"✓ {_pd['brand']} {_pd['name']} aggiunto con rating {_pd['rating']}!")
                st.session_state.pop('pending_duplicate', None)
                st.rerun()
        with _dc2:
            if st.button("❌ Annulla", key="cancel_add_btn"):
                st.session_state.pop('pending_duplicate', None)
                st.rerun()

    gtab_aggiungi, gtab_match, gtab_arricchisci = st.tabs([
        "➕ Aggiungi",
        "🔗 Abbina note",
        "🔍 Arricchisci note",
    ])

    # ─── Gestione → Aggiungi ──────────────────────────────────────────────────
    with gtab_aggiungi:
        with st.form("add_fragrance"):
            col1, col2 = st.columns(2)
            with col1:
                new_brand = st.text_input("Brand *")
                new_name  = st.text_input("Nome *")
                new_form  = st.selectbox("Formato",
                                         ["EdP", "EdT", "Extrait", "Parfum", "EdC", ""])
            with col2:
                new_status = st.selectbox(
                    "Status",
                    ["", "Bottle", "Decant", "Sample", "Tested"],
                    format_func=lambda x: x or "— Nessuno —",
                    help="Bottle=possiedi la bottiglia · Decant/Sample=formato ridotto · "
                         "Tested=provato non posseduto"
                )
                new_watchlist = st.checkbox(
                    "Watchlist",
                    help="Segnalibro, disponibile con qualsiasi status"
                )
                new_bottle = st.checkbox(
                    "Bottle candidate",
                    help="Ignorato se Status = Bottle (già posseduto)"
                )
                new_rating = st.select_slider(
                    "Rating *",
                    options=[r/10 for r in range(
                        int(MIN_RATING*10),
                        int(MAX_RATING*10)+1,
                        int(RATING_STEP*10)
                    )]
                )

            new_comment = st.text_area(
                "Commento",
                placeholder="Note personali, impressioni, occasioni d'uso…"
            )

            submitted = st.form_submit_button("➕ Aggiungi", type="primary")

            if submitted:
                if not new_brand or not new_name:
                    st.error("Brand e nome sono obbligatori.")
                elif check_duplicate(new_brand, new_name):
                    st.session_state['pending_duplicate'] = {
                        'brand':            new_brand,
                        'name':             new_name,
                        'form':             new_form,
                        'status':           new_status,
                        'watchlist':        new_watchlist,
                        'rating':           new_rating,
                        'comment':          new_comment,
                        'bottle_candidate': new_bottle,
                    }
                    st.rerun()
                else:
                    add_rating(
                        brand=new_brand,
                        name=new_name,
                        form=new_form,
                        status=new_status,
                        watchlist=new_watchlist,
                        rating=new_rating,
                        comment=new_comment,
                        bottle_candidate=new_bottle,
                    )
                    st.success(
                        f"✓ {new_brand} {new_name} aggiunto con rating {new_rating}!"
                    )

    # ─── Gestione → Abbina note ───────────────────────────────────────────────
    with gtab_match:
        st.caption("Collega ogni tuo profumo al dataset Kaggle per ottenere le note olfattive.")

        df_ratings = load_ratings()
        confirmed  = load_confirmed_matches()

        if df_ratings.empty:
            st.info("Nessuna fragranza nella lista.")
        else:
            n_total    = len(df_ratings)
            n_confirmed = len([v for v in confirmed.values() if v is not None])
            n_skipped   = len([v for v in confirmed.values() if v is None])
            n_pending   = n_total - len(confirmed)

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Totale",   n_total)
            col2.metric("Abbinate", n_confirmed)
            col3.metric("Saltate",  n_skipped)
            col4.metric("Da fare",  n_pending)

            st.divider()

            with st.expander("🔍 Cerca manualmente nel dataset"):
                search_query = st.text_input(
                    "Cerca per nome o brand",
                    placeholder="es. philosykos, diptyque, fig..."
                )
                if search_query:
                    mask = (
                        df_dataset['name_clean'].str.contains(
                            search_query.lower(), na=False) |
                        df_dataset['brand_clean'].str.contains(
                            search_query.lower(), na=False)
                    )
                    df_search = df_dataset[mask].head(100)
                    if df_search.empty:
                        st.caption("Nessun risultato.")
                    else:
                        my_options = df_ratings.apply(
                            lambda r: (
                                f"{r['brand']} — {r['name']} {r.get('form', '')} "
                                f"({str(r.get('status', '') or '—')})"
                            ),
                            axis=1
                        ).tolist()
                        target = st.selectbox("Abbina a quale tuo profumo?",
                                               my_options, key="manual_target")
                        target_label = target.split(' (')[0]
                        matches = df_ratings[
                            df_ratings.apply(
                                lambda r: f"{r['brand']} — {r['name']} {r.get('form', '')}",
                                axis=1
                            ).str.strip() == target_label.strip()
                        ]
                        if matches.empty:
                            st.error("Profumo non trovato nella lista.")
                            st.stop()
                        target_idx = matches.index[0]

                        for _, sr in df_search.iterrows():
                            col1, col2 = st.columns([4, 1])
                            with col1:
                                st.caption(
                                    f"**{sr['Brand']} — {sr['Perfume']}** | "
                                    f"Top: {sr.get('Top', '—')} | "
                                    f"Middle: {sr.get('Middle', '—')} | "
                                    f"Base: {sr.get('Base', '—')}"
                                )
                            with col2:
                                if st.button("✓ Usa", key=f"manual_{sr.name}"):
                                    confirmed[target_idx] = (
                                        int(sr.name),
                                        str(sr['Brand']),
                                        str(sr['Perfume']),
                                    )
                                    save_confirmed_matches(confirmed)
                                    df_enriched = enrich_ratings(df_ratings, confirmed)
                                    save_ratings(df_enriched)
                                    st.success("Abbinato!")
                                    st.rerun()

            st.divider()

            show_pending = st.checkbox("Mostra solo da abbinare", value=True)

            for idx, row in df_ratings.iterrows():
                if show_pending and idx in confirmed:
                    continue

                already = confirmed.get(idx)
                _match_status = (
                    "✅" if (idx in confirmed and already is not None) else
                    "⏭️" if (idx in confirmed and already is None) else
                    "⏳"
                )

                with st.expander(
                    f"{_match_status} {row['brand']} — {row['name']} {row.get('form', '')}"
                ):
                    st.caption(f"Il tuo voto: {row['rating']} | {row.get('comment', '')}")

                    candidates = get_candidates(
                        row['brand'], row['name'], df_dataset, n=15, threshold=35
                    )

                    if not candidates:
                        st.warning("Nessun candidato trovato nel dataset.")
                        if st.button("Segna come non trovato",
                                     key=f"match_skip_{idx}"):
                            confirmed[idx] = None
                            save_confirmed_matches(confirmed)
                            st.rerun()
                    else:
                        st.write("Seleziona il profumo corretto:")
                        for c in candidates:
                            col1, col2 = st.columns([3, 1])
                            with col1:
                                st.write(f"**{c['brand']} — {c['name']}** "
                                         f"(score: {c['score']})")
                                if c['top']:
                                    st.caption(f"Top: {c['top']}")
                                if c['middle']:
                                    st.caption(f"Middle: {c['middle']}")
                                if c['base']:
                                    st.caption(f"Base: {c['base']}")
                                if c['accords']:
                                    st.caption(f"Accords: {c['accords']}")
                            with col2:
                                if st.button("✓ Questo",
                                             key=f"match_{idx}_{c['dataset_idx']}"):
                                    confirmed[idx] = (
                                        c['dataset_idx'],
                                        c['brand'],
                                        c['name'],
                                    )
                                    save_confirmed_matches(confirmed)
                                    df_enriched = enrich_ratings(df_ratings, confirmed)
                                    save_ratings(df_enriched)
                                    st.success("Abbinato!")
                                    st.rerun()

                        if st.button("⏭️ Nessuno di questi", key=f"none_{idx}"):
                            confirmed[idx] = None
                            save_confirmed_matches(confirmed)
                            st.rerun()

    # ─── Gestione → Arricchisci note ─────────────────────────────────────────
    with gtab_arricchisci:
        st.caption("Cerca online le note olfattive per i profumi senza piramide.")

        df_ratings = load_ratings()

        _matched = df_ratings.get('matched',    pd.Series(False, index=df_ratings.index)).fillna(False).astype(bool)
        _top     = df_ratings.get('top_notes',  pd.Series('',    index=df_ratings.index)).fillna('')
        _middle  = df_ratings.get('middle_notes', pd.Series('',  index=df_ratings.index)).fillna('')
        _base    = df_ratings.get('base_notes', pd.Series('',    index=df_ratings.index)).fillna('')

        mask = (~_matched) | (_matched & (_top == '') & (_middle == '') & (_base == ''))
        df_to_enrich = df_ratings[mask]

        if df_to_enrich.empty:
            st.success("Tutti i profumi hanno già le note olfattive!")
        else:
            st.info(f"{len(df_to_enrich)} profumi senza note olfattive complete.")

            for idx, row in df_to_enrich.iterrows():
                with st.expander(
                    f"**{row['brand']} — {row['name']}** {row.get('form', '')}"
                ):
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        st.caption(f"Rating: {row['rating']} | {row.get('comment', '')}")
                        st.caption(
                            "✅ Matchato nel dataset (note mancanti)"
                            if row.get('matched', False)
                            else "⚠️ Non matchato nel dataset"
                        )
                    with col2:
                        search_clicked = st.button("🔍 Cerca online",
                                                    key=f"search_{idx}")

                    result_key = f"enrich_result_{idx}"

                    if search_clicked:
                        with st.spinner("Cerco su Fragrantica…"):
                            result = enrich_from_web(row['brand'], row['name'])
                        st.session_state[result_key] = result if result else 'not_found'

                    if result_key in st.session_state:
                        result = st.session_state[result_key]

                        if result == 'not_found':
                            st.warning("Nessun risultato trovato su Fragrantica.")
                        else:
                            st.divider()

                            url_input = st.text_input(
                                "URL Fragrantica",
                                value=result['url'],
                                key=f"url_input_{idx}",
                                help="Puoi modificare l'URL se il profumo trovato non è quello corretto"
                            )

                            if url_input != result['url']:
                                if st.button("🔄 Ricarica con nuovo URL",
                                             key=f"reload_{idx}"):
                                    with st.spinner("Carico…"):
                                        new_result = scrape_fragrantica(url_input)
                                    if new_result:
                                        new_result['url'] = url_input
                                        st.session_state[result_key] = new_result
                                        st.rerun()
                                    else:
                                        st.error("Nessuna nota trovata a quell'URL.")

                            col_a, col_b = st.columns(2)
                            with col_a:
                                st.markdown("**Note trovate:**")
                                if result['top']:
                                    st.caption(f"Top: {result['top']}")
                                if result['middle']:
                                    st.caption(f"Middle: {result['middle']}")
                                if result['base']:
                                    st.caption(f"Base: {result['base']}")
                                if not any([result['top'], result['middle'], result['base']]):
                                    st.caption("Nessuna piramide disponibile.")
                            with col_b:
                                st.markdown("**Accords:**")
                                st.caption(result['accords'] or '—')

                            st.divider()
                            col_save, col_skip = st.columns(2)
                            with col_save:
                                if st.button("✅ Salva", key=f"save_{idx}",
                                             type="primary"):
                                    save_enriched_notes(
                                        idx=idx,
                                        top=result['top'],
                                        middle=result['middle'],
                                        base=result['base'],
                                        accords=result['accords'],
                                    )
                                    del st.session_state[result_key]
                                    st.success("Note salvate!")
                                    st.rerun()
                            with col_skip:
                                if st.button("⏭️ Salta", key=f"enrich_skip_{idx}"):
                                    del st.session_state[result_key]
                                    st.rerun()
