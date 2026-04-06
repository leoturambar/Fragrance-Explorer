import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from config import ACCORDS, EXPLORATION_STYLES, GENDER_OPTIONS, MIN_RATING, MAX_RATING, RATING_STEP
from enricher import enrich_from_web, scrape_fragrantica
from parser import load_ratings, save_ratings, add_rating, save_enriched_notes, delete_rating
from matcher import load_dataset, get_candidates, enrich_ratings, save_confirmed_matches, load_confirmed_matches
from recommender import get_recommendations, get_similar_to, get_exploration_recommendations, build_personal_profile, compute_scatter_data
from llm import explain_recommendation, explain_profile, explain_exploration

# ── Setup ─────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Fragrance Explorer",
    page_icon="🌿",
    layout="wide"
)

# ── Cache dataset Kaggle (caricato una volta sola) ────────────────────────────

@st.cache_data
def load_dataset_cached():
    return load_dataset()

df_dataset = load_dataset_cached()

# ── Tab structure ─────────────────────────────────────────────────────────────

tab_lista, tab_match, tab_profilo, tab_raccomanda, tab_esplora, tab_aggiungi, tab_arricchisci = st.tabs([
    "📋 La mia lista",
    "🔗 Abbina note",
    "🧬 Profilo olfattivo",
    "✨ Raccomandazioni",
    "🧭 Esplora stili",
    "➕ Aggiungi",
    "🔍 Arricchisci note"
])

with tab_lista:
    st.header("La mia lista")

    df_ratings = load_ratings()

    if df_ratings.empty:
        st.info("Nessuna fragranza ancora. Usa il tab Aggiungi o importa la tua lista HTML.")
    else:
        # ── Filtri ────────────────────────────────────────────────────────
        col1, col2, col3 = st.columns(3)
        with col1:
            rating_min = st.slider("Rating minimo", MIN_RATING, MAX_RATING,
                                   MIN_RATING, RATING_STEP)
        with col2:
            only_bottles = st.checkbox("Solo bottle candidates")
        with col3:
            sort_by = st.selectbox("Ordina per", ["Rating ↓", "Rating ↑", "Brand A-Z"])

        df_view = df_ratings.copy()
        if only_bottles:
            df_view = df_view[df_view['bottle_candidate'] == True]
        df_view = df_view[df_view['rating'] >= rating_min]

        if sort_by == "Rating ↓":
            df_view = df_view.sort_values('rating', ascending=False)
        elif sort_by == "Rating ↑":
            df_view = df_view.sort_values('rating', ascending=True)
        else:
            df_view = df_view.sort_values('brand')

        st.caption(f"{len(df_view)} fragranze")

        # ── Lista ─────────────────────────────────────────────────────────
        for idx, row in df_view.iterrows():
            with st.expander(
                f"**{row['rating']}** — {row['brand']} {row['name']} "
                f"{'⭐' if row.get('bottle_candidate') else ''}"
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
                    st.caption(f"Possesso: {row.get('ownership', '—')}")
                    matched = row.get('matched', False)
                    if matched:
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
                        e_ownership = st.selectbox(
                            "Possesso",
                            ["Decant", "Full Bottle", "Sample", "Wishlist"],
                            index=["Decant", "Full Bottle", "Sample",
                                   "Wishlist"].index(row.get('ownership', 'Decant'))
                            if row.get('ownership') in
                            ["Decant", "Full Bottle", "Sample", "Wishlist"] else 0
                        )
                        e_bottle = st.checkbox(
                            "Bottle candidate",
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
                            df_ratings.at[idx, 'ownership'] = e_ownership
                            df_ratings.at[idx, 'bottle_candidate'] = e_bottle
                            save_ratings(df_ratings)
                            st.session_state[f"editing_{idx}"] = False
                            st.success("Modificato!")
                            st.rerun()

                        if cancel_edit:
                            st.session_state[f"editing_{idx}"] = False
                            st.rerun()

with tab_match:
    confirmed = load_confirmed_matches()
    st.header("Abbina note")
    st.caption("Collega ogni tuo profumo al dataset Kaggle per ottenere le note olfattive.")

    df_ratings = load_ratings()
    confirmed = load_confirmed_matches()

    if df_ratings.empty:
        st.info("Nessuna fragranza nella lista.")
    else:
        # statistiche matching
        n_total = len(df_ratings)
        n_confirmed = len([v for v in confirmed.values() if v is not None])
        n_skipped = len([v for v in confirmed.values() if v is None])
        n_pending = n_total - len(confirmed)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Totale", n_total)
        col2.metric("Abbinate", n_confirmed)
        col3.metric("Saltate", n_skipped)
        col4.metric("Da fare", n_pending)

        st.divider()

        # ── Ricerca manuale nel dataset ───────────────────────────────────
        with st.expander("🔍 Cerca manualmente nel dataset"):
            search_query = st.text_input("Cerca per nome o brand",
                                          placeholder="es. philosykos, diptyque, fig...")
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
                    # seleziona a quale profumo della tua lista abbinare
                    my_options = df_ratings.apply(
                        lambda r: f"{r['brand']} — {r['name']} {r.get('form', '')} ({r.get('ownership', '')})",
                        axis=1
                    ).tolist()
                    target = st.selectbox("Abbina a quale tuo profumo?",
                                          my_options, key="manual_target")
                    target_label = target.split(' (')[0]  # rimuove la parte " (Decant)" ecc.
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
                            if st.button("✓ Usa",
                                         key=f"manual_{sr.name}"):
                                confirmed[target_idx] = int(sr.name)
                                save_confirmed_matches(confirmed)
                                df_enriched = enrich_ratings(df_ratings, confirmed)
                                save_ratings(df_enriched)
                                st.success("Abbinato!")
                                st.rerun()

        st.divider()

        # mostra solo profumi non ancora processati
        show_pending = st.checkbox("Mostra solo da abbinare", value=True)

        for idx, row in df_ratings.iterrows():
            if show_pending and idx in confirmed:
                continue

            already = confirmed.get(idx)
            status = "✅" if (idx in confirmed and already is not None) else \
                     "⏭️" if (idx in confirmed and already is None) else "⏳"

            with st.expander(
                f"{status} {row['brand']} — {row['name']} {row.get('form', '')}"
            ):
                st.caption(f"Il tuo voto: {row['rating']} | {row.get('comment', '')}")

                candidates = get_candidates(row['brand'], row['name'], df_dataset, n=15, threshold=35)

                if not candidates:
                    st.warning("Nessun candidato trovato nel dataset.")
                    if st.button("Segna come non trovato", key=f"skip_{idx}"):
                        confirmed[idx] = None
                        save_confirmed_matches(confirmed)
                        st.rerun()
                else:
                    st.write("Seleziona il profumo corretto:")
                    for c in candidates:
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.write(f"**{c['brand']} — {c['name']}** (score: {c['score']})")
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
                                confirmed[idx] = c['dataset_idx']
                                save_confirmed_matches(confirmed)
                                # aggiorna CSV ratings con le note
                                df_enriched = enrich_ratings(df_ratings, confirmed)
                                save_ratings(df_enriched)
                                st.success("Abbinato!")
                                st.rerun()

                    if st.button("⏭️ Nessuno di questi",
                                 key=f"none_{idx}"):
                        confirmed[idx] = None
                        save_confirmed_matches(confirmed)
                        st.rerun()

with tab_profilo:
    st.header("Profilo olfattivo")

    df_ratings = load_ratings()

    if df_ratings.empty:
        st.info("Nessuna fragranza nella lista.")
    else:
        # ── Pre-calcolo top_accords (serve sia per metric che per radar) ──────
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

        # ── Statistiche — full width ──────────────────────────────────────────
        st.subheader("Statistiche")

        total        = len(df_ratings)
        avg_rating   = df_ratings['rating'].mean()
        best_row     = df_ratings.loc[df_ratings['rating'].idxmax()]
        full_bottles = len(df_ratings[df_ratings['ownership'] == 'Full Bottle'])
        decants      = len(df_ratings[df_ratings['ownership'] == 'Decant'])
        bottle_candidates = len(df_ratings[df_ratings['bottle_candidate'] == True])
        fam_preferita = top_accords[0][0].capitalize() if top_accords else '—'

        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Totale",             total)
        c2.metric("Avg rating",         f"{avg_rating:.1f}")
        c3.metric("Full bottles",       full_bottles)
        c4.metric("Decants",            decants)
        c5.metric("Bottle Candidates",  bottle_candidates)
        c6.metric("Famiglia preferita", fam_preferita)

        st.divider()

        # ── Bar chart + Radar — due colonne ───────────────────────────────────
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
                st.info("Abbina le note nel tab 'Abbina note' per vedere il radar.")
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

        # ── Scatter plot — full width ─────────────────────────────────────────
        st.subheader("La tua collezione nello spazio olfattivo")
        df_scatter = compute_scatter_data(df_ratings)

        if df_scatter.empty:
            st.caption("Abbina qualche profumo per vedere il grafico.")
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
                legend=dict(
                    orientation='h',
                    x=0, y=1.08,
                    xanchor='left',
                    yanchor='bottom',
                ),
                height=500,
                margin=dict(l=40, r=40, t=60, b=40),
                plot_bgcolor='rgba(240,240,240,0.1)',
            )

            st.plotly_chart(fig_scatter, width='stretch', key="scatter_collezione")

        st.divider()

        # ── Analisi LLM — full width ──────────────────────────────────────────
        st.subheader("Analisi del tuo profilo")
        if st.button("🤖 Genera analisi profilo", type="primary"):
            with st.spinner("Analisi in corso…"):
                try:
                    analysis = explain_profile(df_ratings)
                    st.markdown(analysis)
                except Exception as e:
                    st.error(f"Errore: {e}")


with tab_raccomanda:
    st.header("Raccomandazioni")

    df_ratings = load_ratings()

    if 'matched' not in df_ratings.columns or df_ratings['matched'].sum() == 0:
        st.info("Abbina almeno qualche profumo nel tab 'Abbina note' per ottenere raccomandazioni.")
    else:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            n_recs = st.slider("Numero raccomandazioni", 5, 20, 10)
        with col2:
            gender_label = st.selectbox("Genere", list(GENDER_OPTIONS.keys()))
            gender_val = GENDER_OPTIONS[gender_label]
        with col3:
            exclude_known = st.checkbox("Escludi profumi già noti", value=True)
        with col4:
            quality_pct = st.slider(
                "Filtro qualità",
                10, 100, 40, 10,
                help="10% = solo profumi top | 40% = equilibrato | 100% = tutti"
            )

        st.divider()

        mode = st.radio(
            "Modalità",
            options=['profilo', 'simile'],
            format_func=lambda x: {
                'profilo': '🧬 Basato sul mio profilo',
                'simile':  '🔍 Simile a un profumo specifico',
            }[x],
            horizontal=True
        )

        if mode == 'simile':
            known = df_ratings[['brand', 'name']].apply(
                lambda r: f"{r['brand']} — {r['name']}", axis=1
            ).tolist()
            selected = st.selectbox("Scegli profumo di riferimento", known)
            ref_brand, ref_name = selected.split(' — ', 1)

        if st.button("✨ Genera raccomandazioni", type="primary"):
            with st.spinner("Calcolo in corso…"):
                if mode == 'profilo':
                    df_recs = get_recommendations(
                    df_ratings, df_dataset, n=n_recs,
                    exclude_known=exclude_known,
                    gender_filter=gender_val,
                    quality_pct=quality_pct
                )
                else:
                    df_recs = get_similar_to(
                        ref_brand, ref_name,
                        df_ratings, df_dataset, n=n_recs,
                        quality_pct=quality_pct
                    )

            if df_recs.empty:
                st.warning("Nessuna raccomandazione trovata.")
            else:
                for _, rec in df_recs.iterrows():
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

                        if st.button("🤖 Spiega perché", key=f"explain_{rec['brand']}_{rec['name']}"):
                            with st.spinner("Analisi in corso…"):
                                try:
                                    explanation = explain_recommendation(
                                        rec.to_dict(), df_ratings
                                    )
                                    st.markdown(explanation)
                                except Exception as e:
                                    st.error(f"Errore: {e}")

with tab_esplora:
    st.header("Esplora stili")
    st.caption("Scopri profumi in territori olfattivi nuovi, ancorati al tuo gusto.")

    df_ratings = load_ratings()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        style_label = st.selectbox("Stile da esplorare",
                                list(EXPLORATION_STYLES.keys()))
    with col2:
        intensity = st.slider(
            "Intensità esplorazione",
            0.0, 1.0, 0.5, 0.1,
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

    if st.button("🧭 Esplora", type="primary"):
        with st.spinner("Calcolo in corso…"):
            style_query = EXPLORATION_STYLES[style_label]
            df_exp = get_exploration_recommendations(
                df_ratings, df_dataset,
                explore_style=style_query,
                intensity=intensity,
                n=10,
                gender_filter=gender_val_e,
                quality_pct=quality_pct_e
            )

        if df_exp.empty:
            st.warning("Nessun risultato trovato.")
        else:
            # spiegazione LLM
            if 'matched' in df_ratings.columns and df_ratings['matched'].sum() > 0:
                with st.spinner("Generando spiegazione…"):
                    try:
                        explanation = explain_exploration(
                            style_label, df_ratings, df_exp)
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

with tab_aggiungi:
    st.header("Aggiungi fragranza")

    with st.form("add_fragrance"):
        col1, col2 = st.columns(2)
        with col1:
            new_brand = st.text_input("Brand *")
            new_name = st.text_input("Nome *")
            new_form = st.selectbox("Formato", ["EdP", "EdT", "Extrait", "Parfum", "EdC", ""])
        with col2:
            new_ownership = st.selectbox("Possesso",
                ["Decant", "Full Bottle", "Sample", "Wishlist"])
            new_rating = st.select_slider(
                "Rating *",
                options=[r/10 for r in range(
                    int(MIN_RATING*10),
                    int(MAX_RATING*10)+1,
                    int(RATING_STEP*10)
                )]
            )
            new_bottle = st.checkbox("Bottle candidate")

        new_comment = st.text_area("Commento", placeholder="Note personali, impressioni, occasioni d'uso…")

        submitted = st.form_submit_button("➕ Aggiungi", type="primary")

        if submitted:
            if not new_brand or not new_name:
                st.error("Brand e nome sono obbligatori.")
            else:
                add_rating(
                    brand=new_brand,
                    name=new_name,
                    form=new_form,
                    ownership=new_ownership,
                    rating=new_rating,
                    comment=new_comment,
                    bottle_candidate=new_bottle
                )
                st.success(f"✓ {new_brand} {new_name} aggiunto con rating {new_rating}!")


with tab_arricchisci:
    st.header("Arricchisci note")
    st.caption("Cerca online le note olfattive per i profumi senza piramide.")

    df_ratings = load_ratings()

    # filtra: non matchati O matchati ma senza note
    matched = df_ratings.get('matched', pd.Series(False, index=df_ratings.index)).fillna(False).astype(bool)
    top     = df_ratings.get('top_notes', pd.Series('', index=df_ratings.index)).fillna('')
    middle  = df_ratings.get('middle_notes', pd.Series('', index=df_ratings.index)).fillna('')
    base    = df_ratings.get('base_notes', pd.Series('', index=df_ratings.index)).fillna('')

    mask = (~matched) | (matched & (top == '') & (middle == '') & (base == ''))
    df_to_enrich = df_ratings[mask]

    if df_to_enrich.empty:
        st.success("Tutti i profumi hanno già le note olfattive!")
    else:
        st.info(f"{len(df_to_enrich)} profumi senza note olfattive complete.")

        for idx, row in df_to_enrich.iterrows():
            with st.expander(f"**{row['brand']} — {row['name']}** {row.get('form', '')}"):

                col1, col2 = st.columns([2, 1])
                with col1:
                    st.caption(f"Rating: {row['rating']} | {row.get('comment', '')}")
                    matched = row.get('matched', False)
                    st.caption("✅ Matchato nel dataset (note mancanti)" if matched
                               else "⚠️ Non matchato nel dataset")
                with col2:
                    search_clicked = st.button(
                        "🔍 Cerca online",
                        key=f"search_{idx}"
                    )

                # risultati della ricerca — persistiti in session_state
                result_key = f"enrich_result_{idx}"

                if search_clicked:
                    with st.spinner(f"Cerco su Fragrantica…"):
                        result = enrich_from_web(row['brand'], row['name'])
                    if result:
                        st.session_state[result_key] = result
                    else:
                        st.session_state[result_key] = 'not_found'

                # mostra risultato se presente
                if result_key in st.session_state:
                    result = st.session_state[result_key]

                    if result == 'not_found':
                        st.warning("Nessun risultato trovato su Fragrantica.")
                    else:
                        st.divider()

                        # URL modificabile — precompilato con quello trovato
                        url_input = st.text_input(
                            "URL Fragrantica",
                            value=result['url'],
                            key=f"url_input_{idx}",
                            help="Puoi modificare l'URL se il profumo trovato non è quello corretto"
                        )

                        # se l'URL è stato cambiato, offri di ricaricare
                        if url_input != result['url']:
                            if st.button("🔄 Ricarica con nuovo URL", key=f"reload_{idx}"):
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
                            if st.button("✅ Salva", key=f"save_{idx}", type="primary"):
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
                            if st.button("⏭️ Salta", key=f"skip_{idx}"):
                                del st.session_state[result_key]
                                st.rerun()