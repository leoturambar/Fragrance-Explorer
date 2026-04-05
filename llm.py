import os
import pandas as pd
from recommender import build_personal_profile

LLM_BACKEND = "ollama"  # "claude" oppure "ollama"

OLLAMA_MODEL  = "qwen2.5:14b"
OLLAMA_BASE_URL = "http://localhost:11434/v1"
CLAUDE_MODEL  = "claude-haiku-4-5-20251001"


def get_client():
    if LLM_BACKEND == "claude":
        import anthropic
        return anthropic.Anthropic()
    else:
        from openai import OpenAI
        return OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")


def call_llm(prompt: str, max_tokens: int = 500) -> str:
    if LLM_BACKEND == "claude":
        client = get_client()
        msg = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}]
        )
        return msg.content[0].text
    else:
        client = get_client()
        resp = client.chat.completions.create(
            model=OLLAMA_MODEL,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}]
        )
        return resp.choices[0].message.content


def explain_recommendation(candidate: dict, df_ratings: pd.DataFrame,
                            user_name: str = "the user") -> str:
    """
    Spiega perché un profumo potrebbe piacere basandosi sul profilo personale.
    candidate: dict con brand, name, top, middle, base, accord1, accord2, similarity
    """
    profile = build_personal_profile(df_ratings)

    # trova i profumi più amati come riferimento
    top_frags = df_ratings[df_ratings['rating'] >= 8.0][
        ['brand', 'name', 'rating', 'comment']
    ].sort_values('rating', ascending=False).head(5)

    top_frags_str = '\n'.join(
        f"  - {r['brand']} {r['name']} ({r['rating']}/10): {r['comment']}"
        for _, r in top_frags.iterrows()
    )

    prompt = f"""You are an expert fragrance consultant with deep knowledge of perfumery.

The user's top-rated fragrances are:
{top_frags_str}

Their olfactory profile (extracted from ratings) tends toward:
{profile['top_notes'][:300] if profile['top_notes'] else 'not enough data yet'}

You are recommending: {candidate['brand']} — {candidate['name']}
Notes: Top: {candidate.get('top', 'n/a')} | Middle: {candidate.get('middle', 'n/a')} | Base: {candidate.get('base', 'n/a')}
Main accords: {candidate.get('accord1', '')}, {candidate.get('accord2', '')}
Similarity score: {candidate.get('similarity', 'n/a')}

In 3-4 sentences, explain specifically why this fragrance might appeal to this user.
Reference their actual favorite fragrances by name. Be specific about which notes create the connection.
Do not use generic phrases like "you might enjoy this". Be precise and direct."""

    return call_llm(prompt, max_tokens=300)


def explain_profile(df_ratings: pd.DataFrame) -> str:
    """
    Genera una descrizione narrativa del profilo olfattivo personale.
    """
    df_matched = df_ratings[df_ratings['matched'] == True]
    if df_matched.empty:
        return "Not enough matched fragrances to build a profile yet."

    top = df_ratings[df_ratings['rating'] >= 8.0][['brand', 'name', 'rating', 'comment']]
    low = df_ratings[df_ratings['rating'] <= 6.0][['brand', 'name', 'rating', 'comment']]

    top_str = '\n'.join(f"  {r['brand']} {r['name']} ({r['rating']}): {r['comment']}"
                        for _, r in top.iterrows())
    low_str = '\n'.join(f"  {r['brand']} {r['name']} ({r['rating']}): {r['comment']}"
                        for _, r in low.iterrows())

    prompt = f"""You are an expert fragrance consultant.

Based on these ratings, describe this person's olfactory profile in a precise, insightful paragraph.
Identify patterns, recurring preferences, what they clearly avoid, and their overall aesthetic.
Use perfumery terminology but keep it accessible.

HIGH-RATED fragrances (loved):
{top_str}

LOW-RATED fragrances (disliked or indifferent):
{low_str}

Write a 150-200 word profile. Be specific, reference actual fragrances and notes.
End with one sentence suggesting what unexplored territory might suit them."""

    return call_llm(prompt, max_tokens=400)


def explain_exploration(style: str, df_ratings: pd.DataFrame,
                        candidates: pd.DataFrame) -> str:
    """
    Spiega perché i profumi suggeriti in modalità esplorazione
    sono un buon punto di ingresso per un nuovo stile.
    """
    top_frags = df_ratings[df_ratings['rating'] >= 8.0][
        ['brand', 'name', 'comment']].head(4)
    top_str = '\n'.join(f"  {r['brand']} {r['name']}: {r['comment']}"
                        for _, r in top_frags.iterrows())

    cands_str = '\n'.join(
        f"  {r['brand']} {r['name']} — {r.get('accord1', '')} / top: {r.get('top', '')}"
        for _, r in candidates.head(4).iterrows()
    )

    prompt = f"""You are an expert fragrance consultant helping someone explore a new style.

The user wants to explore: {style}

Their current favorites:
{top_str}

Recommended entry points into {style}:
{cands_str}

In 3-4 sentences, explain why these specific fragrances are good entry points into {style}
for someone with this taste profile. Mention the bridges between what they already love
and the new style. Be specific."""

    return call_llm(prompt, max_tokens=300)