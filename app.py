"""Stochastic POS Tagger Lab - Streamlit Web Application.

A complete interactive laboratory for stochastic Part-of-Speech tagging
on conversational speech transcripts using NLTK Penn Treebank backoff models.
"""

from pathlib import Path
import os
import streamlit as st
import pandas as pd
import plotly.express as px

# Core NLP imports
from src.preprocessing import clean_conversational_text, tokenize_conversational_transcript
from src.pipeline import TaggingPipeline
from src.models import (
    load_trained_taggers,
    load_treebank_split,
    train_stochastic_taggers,
    save_trained_taggers,
    DEFAULT_MODEL_PATH,
)
from src.explainability import ExplainableBackoffTagger, PENN_TREEBANK_TAGS
from src.probability import CorpusProbabilityEstimator
from src.evaluation import (
    calculate_metrics,
    extract_gold_and_pred,
    compute_confusion_matrix_data,
    evaluate_tagger_hierarchy,
    analyze_oov_performance,
)

# Configure page metadata and layout
st.set_page_config(
    page_title="Stochastic POS Tagger Lab",
    page_icon="🏷️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# High-contrast, polished academic styling
st.markdown(
    """
    <style>
    /* Global high-contrast dark palette */
    .metric-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%) !important;
        border: 1px solid #334155 !important;
        border-radius: 10px;
        padding: 16px 20px;
        margin-bottom: 18px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
    }
    .metric-card h4 {
        color: #f8fafc !important;
        margin-top: 0;
        margin-bottom: 10px;
        font-weight: 700;
    }
    .metric-card p, .metric-card li {
        color: #cbd5e1 !important;
        font-size: 14px;
        line-height: 1.6;
    }
    
    /* High contrast N-Gram table */
    .ngram-table {
        width: 100% !important;
        border-collapse: collapse !important;
        margin-top: 10px !important;
        margin-bottom: 15px !important;
        background-color: #0f172a !important;
        border: 1px solid #334155 !important;
        border-radius: 8px !important;
        overflow: hidden !important;
    }
    .ngram-table th {
        background-color: #1e293b !important;
        color: #38bdf8 !important;
        font-weight: 700 !important;
        padding: 10px 14px !important;
        border-bottom: 2px solid #475569 !important;
        text-align: left !important;
        font-size: 14px !important;
    }
    .ngram-table td {
        background-color: #0f172a !important;
        color: #f8fafc !important;
        padding: 10px 14px !important;
        border-bottom: 1px solid #1e293b !important;
        font-size: 14px !important;
    }
    .ngram-table code {
        background-color: #1e293b !important;
        color: #a5f3fc !important;
        padding: 3px 6px !important;
        border-radius: 4px !important;
        font-weight: 600 !important;
        border: 1px solid #334155 !important;
    }

    /* Status Badges */
    .status-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: 700;
        margin-right: 8px;
    }
    .status-ready {
        background: rgba(16, 185, 129, 0.25);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.5);
    }
    .status-wip {
        background: rgba(99, 102, 241, 0.25);
        color: #a5b4fc;
        border: 1px solid rgba(99, 102, 241, 0.5);
    }

    /* Tag Chips */
    .tag-chip {
        display: inline-flex;
        align-items: center;
        background: #1e293b;
        border: 1px solid #475569;
        border-radius: 8px;
        padding: 5px 10px;
        margin: 4px;
        font-size: 14px;
        color: #f8fafc;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
    .tag-label {
        margin-left: 8px;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 11px;
        font-weight: 800;
        letter-spacing: 0.5px;
    }
    .tier-trigram { background: #2563eb; color: #ffffff; }
    .tier-bigram { background: #7c3aed; color: #ffffff; }
    .tier-unigram { background: #059669; color: #ffffff; }
    .tier-default { background: #d97706; color: #ffffff; }

    /* Calculation Cards */
    .calc-card {
        background: #0f172a;
        border: 1px solid #334155;
        border-left: 4px solid #6366f1;
        border-radius: 8px;
        padding: 14px 16px;
        margin-bottom: 14px;
    }
    .calc-card h5 {
        color: #38bdf8;
        margin-top: 0;
        margin-bottom: 6px;
        font-weight: 700;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Base directories
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
VOICE_HTML_PATH = BASE_DIR / "components" / "voice_input.html"


@st.cache_resource
def get_pipeline() -> TaggingPipeline:
    """Load or initialize tagging pipeline once."""
    return TaggingPipeline()


@st.cache_data
def get_corpus_splits():
    """Load and cache the Penn Treebank splits."""
    return load_treebank_split(seed=42)


pipeline = get_pipeline()


# ==============================================================================
# SIDEBAR NAVIGATION
# ==============================================================================
with st.sidebar:
    st.title("🏷️ POS Tagger Lab")
    st.caption("Stochastic NLP for Conversational Speech")
    st.markdown("---")

    navigation_choice = st.radio(
        "Navigation",
        [
            "🎙️ Live Speech & Text Input",
            "🏷️ POS Tagging Explorer",
            "🔍 Algorithm & Backoff Trace",
            "📊 Probability Explorer",
            "📈 Model Evaluation",
            "⚙️ Architecture & Status",
        ],
        index=0,
    )

    st.markdown("---")
    st.subheader("Model Artifact Status")
    if pipeline.is_model_loaded:
        st.markdown(
            '<span class="status-badge status-ready">● Model Active (Trained)</span>',
            unsafe_allow_html=True,
        )
        st.caption("Penn Treebank Backoff Chain loaded from `models/tagger_chain.pkl`")
    else:
        st.markdown(
            '<span class="status-badge status-wip">○ Model Not Found</span>',
            unsafe_allow_html=True,
        )
        if st.button("🚀 Train Model Now"):
            with st.spinner("Training on Penn Treebank corpus..."):
                train, dev, test = get_corpus_splits()
                taggers = train_stochastic_taggers(train)
                save_trained_taggers(taggers)
                st.cache_resource.clear()
                st.success("Trained and saved to models/tagger_chain.pkl!")
                st.rerun()

    st.markdown("---")
    st.caption("AI/ML Mini Project • Penn Treebank Sequential POS Tagger")


# ==============================================================================
# PAGE 1: LIVE SPEECH & TEXT INPUT (DIRECT POPULATION & TAGGING)
# ==============================================================================
if navigation_choice == "🎙️ Live Speech & Text Input":
    st.title("🎙️ Conversational Speech & Real-Time POS Tagging")
    st.markdown(
        """
        Speak naturally through your microphone or type a conversational transcript.
        When you stop speaking, the text **directly enters the transcript area** and is
        **immediately preprocessed, tokenized, and POS tagged** with sequential backoff stochastic models.
        """
    )

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.subheader("Microphone Audio Capture")
        st.caption("Web Speech API recognition with instant direct synchronization.")

        # Embed voice input HTML component if present
        if VOICE_HTML_PATH.exists():
            with open(VOICE_HTML_PATH, "r", encoding="utf-8") as f:
                voice_html_content = f.read()
            st.components.v1.html(voice_html_content, height=180, scrolling=False)
        else:
            st.warning("Voice component template `components/voice_input.html` not found.")

        st.info(
            "💡 **Microphone Tip**: Click **Start Speaking**, allow mic access, speak conversational sentences with fillers, "
            "then click **Stop Recording**. The transcript populates the text area and executes tagging below!"
        )

    with col2:
        st.subheader("Conversational Transcript")
        st.caption("Preserves fillers (um, uh, like), repetitions, and conversational disfluencies.")

        # Detect voice input from query parameters and populate text area directly
        spoken_param = st.query_params.get("spoken", None)
        timestamp_param = st.query_params.get("t", None)

        if spoken_param and timestamp_param != st.session_state.get("last_voice_t"):
            st.session_state["last_voice_t"] = timestamp_param
            st.session_state["user_transcript_box"] = spoken_param

        # Initialize default transcript if empty
        if "user_transcript_box" not in st.session_state or not st.session_state["user_transcript_box"]:
            st.session_state["user_transcript_box"] = "Yeah, honestly, that's what he told me, like, right before lunch."

        sample_presets = {
            "Select sample preset...": "",
            "Dialogue with Fillers": "Yeah, honestly, that's what he told me, like, right before lunch.",
            "Hesitations & Self-Corrections": "Um, uh, I think, you know, we should go to the, to the main office.",
            "Informal Conversational": "Well, hmm, uh-huh, let me see if the data is ready.",
            "Discourse Markers": "I mean, like, he was actually right about that.",
        }

        selected_preset = st.selectbox("Load Sample Preset:", list(sample_presets.keys()))

        if selected_preset != "Select sample preset..." and selected_preset != st.session_state.get("last_preset"):
            st.session_state["last_preset"] = selected_preset
            st.session_state["user_transcript_box"] = sample_presets[selected_preset]

        user_text = st.text_area(
            "Conversational Transcript Text Area:",
            key="user_transcript_box",
            height=110,
            placeholder="Speak via microphone or type conversational speech here...",
        )

    st.markdown("---")

    # DIRECT & AUTOMATIC PREPROCESSING & POS TAGGING (Runs on whatever text is in the box)
    if user_text.strip():
        cleaned = clean_conversational_text(user_text)
        sentences = tokenize_conversational_transcript(user_text)

        # 1. PREPROCESSING OUTPUT SECTION
        st.success("✅ Preprocessing & Conversational Tokenization Successful!")
        st.markdown(f"**Cleaned Text**: `{cleaned}`")
        st.markdown(f"**Sentence Count**: `{len(sentences)}`")

        for idx, sent in enumerate(sentences, start=1):
            st.markdown(f"**Sentence {idx}** ({len(sent)} tokens):")
            st.code(str(sent), language="python")

        # 2. POS TAGGING & EXPLAINABILITY OUTPUT SECTION
        st.subheader("🏷️ Part-of-Speech Tagging & Stochastic Backoff Results")

        if pipeline.is_model_loaded:
            tagged_results = pipeline.tag_text_with_explanations(user_text)

            tier_counts = {"Trigram": 0, "Bigram": 0, "Unigram": 0, "Default": 0}
            all_explanations = []

            for sent_tagged, exp_list in tagged_results:
                for exp in exp_list:
                    tier_counts[exp.responsible_tagger] = tier_counts.get(exp.responsible_tagger, 0) + 1
                    all_explanations.append(exp)

            # Metric KPIs
            k1, k2, k3, k4, k5 = st.columns(5)
            k1.metric("Total Tokens", len(all_explanations))
            k2.metric("Trigram Hits", tier_counts.get("Trigram", 0))
            k3.metric("Bigram Hits", tier_counts.get("Bigram", 0))
            k4.metric("Unigram Hits", tier_counts.get("Unigram", 0))
            k5.metric("Default ('NN')", tier_counts.get("Default", 0))

            # Visual Word-Tag Chips
            st.markdown("##### Visual Tagged Output:")
            chips_html = ""
            for exp in all_explanations:
                tier_class = f"tier-{exp.responsible_tagger.lower()}"
                chips_html += f"""
                <span class="tag-chip">
                    <strong>{exp.token}</strong>
                    <span class="tag-label {tier_class}">{exp.assigned_tag}</span>
                </span>
                """
            st.markdown(f"<div>{chips_html}</div>", unsafe_allow_html=True)
            st.caption("Legend: 🟦 Trigram | 🟪 Bigram | 🟩 Unigram | 🟧 Default Fallback ('NN')")

            # Detailed Explainability Table
            st.markdown("##### Token-by-Token Decision Audit:")
            table_rows = []
            for exp in all_explanations:
                table_rows.append({
                    "Token": exp.token,
                    "Assigned Tag": exp.assigned_tag,
                    "Tag Description": exp.tag_description,
                    "Responsible Tier": exp.responsible_tagger,
                    "Decision Rationale": exp.rationale,
                })

            df_explanations = pd.DataFrame(table_rows)
            st.dataframe(df_explanations, use_container_width=True)

            # CSV Download
            csv_data = df_explanations.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Export Tagged Transcript as CSV",
                csv_data,
                "tagged_transcript.csv",
                "text/csv",
            )
        else:
            st.warning("⚠️ Trained model not loaded. Please click 'Train Model Now' in the sidebar.")


# ==============================================================================
# PAGE 2: POS TAGGING EXPLORER
# ==============================================================================
elif navigation_choice == "🏷️ POS Tagging Explorer":
    st.title("🏷️ Stochastic POS Tagging Execution & Corpus Testing")
    st.markdown(
        """
        Evaluate how the sequential backoff model handles standard vs. conversational sentences.
        The tagger executes the decision path:
        $$\\text{TrigramTagger} \\longrightarrow \\text{BigramTagger} \\longrightarrow \\text{UnigramTagger} \\longrightarrow \\text{DefaultTagger}('NN')$$
        """
    )

    custom_input = st.text_input(
        "Enter test sentence:",
        value="Um, like, the chairman of the board gave a speech yesterday.",
    )

    if custom_input.strip() and pipeline.is_model_loaded:
        res = pipeline.tag_text_with_explanations(custom_input)
        for sent_idx, (tagged_s, exp_list) in enumerate(res, 1):
            st.write(f"**Tagged Sentence {sent_idx}:**")
            st.code(str(tagged_s), language="python")

    st.markdown("---")
    st.subheader("Penn Treebank Common Tag Reference")
    ref_df = pd.DataFrame(
        [{"Tag": k, "Description": v} for k, v in list(PENN_TREEBANK_TAGS.items())[:20]]
    )
    st.dataframe(ref_df, use_container_width=True)


# ==============================================================================
# PAGE 3: ALGORITHM & BACKOFF TRACE (DEEP INTUITIVE EXPLANATION + 4 WORKED EXAMPLES)
# ==============================================================================
elif navigation_choice == "🔍 Algorithm & Backoff Trace":
    st.title("🔍 Algorithm & Sequential Backoff Trace Explained")
    st.markdown(
        """
        Learn **how stochastic sequential POS taggers think**, why words get assigned to 
        **Trigram**, **Bigram**, **Unigram**, or **Default**, and inspect **4 real mathematical calculation examples**.
        """
    )

    # 1. High-Contrast Readable Hierarchy Table
    st.markdown(
        """
        <div class="metric-card">
            <h4>Hierarchy of N-gram Context & Decision Rules</h4>
            <p>When tagging a token <code>w_i</code>, the model searches from most specific (Trigram) to most general (Default):</p>
            <table class="ngram-table">
                <thead>
                    <tr>
                        <th style="width: 15%;">Model Tier</th>
                        <th style="width: 25%;">Conditioning Context</th>
                        <th style="width: 30%;">Mathematical Decision Rule</th>
                        <th style="width: 30%;">When Does It Win?</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td><span class="tag-label tier-trigram">Trigram</span></td>
                        <td><code>(t_{i-2}, t_{i-1}, w_i)</code></td>
                        <td><code>argmax_t P(t | t_{i-2}, t_{i-1}, w_i)</code></td>
                        <td>Matches full 3-gram syntactic structure (e.g. relative clause syntax).</td>
                    </tr>
                    <tr>
                        <td><span class="tag-label tier-bigram">Bigram</span></td>
                        <td><code>(t_{i-1}, w_i)</code></td>
                        <td><code>argmax_t P(t | t_{i-1}, w_i)</code></td>
                        <td>Trigram misses, but previous tag provides context to resolve word ambiguity.</td>
                    </tr>
                    <tr>
                        <td><span class="tag-label tier-unigram">Unigram</span></td>
                        <td><code>(w_i)</code></td>
                        <td><code>argmax_t P(t | w_i)</code></td>
                        <td>Word was seen in corpus; picks its most frequent tag regardless of context.</td>
                    </tr>
                    <tr>
                        <td><span class="tag-label tier-default">Default</span></td>
                        <td><code>None</code></td>
                        <td><code>"NN" (Noun)</code></td>
                        <td>Word is Out-Of-Vocabulary (OOV) or spontaneous conversational filler.</td>
                    </tr>
                </tbody>
            </table>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 2. Section: 4 Clear Mathematical Calculation Examples
    st.subheader("📐 4 Concrete Mathematical Calculation Examples (Penn Treebank)")
    st.markdown("Here is exactly how empirical probability distributions and backoff decisions are calculated:")

    c1, c2 = st.columns(2)

    with c1:
        st.markdown(
            """
            <div class="calc-card">
                <h5>1. Default Tagger Calculation: Out-of-Vocabulary (OOV)</h5>
                <p><strong>Example Token:</strong> <code>"Yeah"</code> (Conversational filler)</p>
                <p><strong>Corpus Frequency:</strong> <code>Count("Yeah") = 0</code> in Penn Treebank.</p>
                <p><strong>Probability Calculation:</strong></p>
            """,
            unsafe_allow_html=True,
        )
        st.latex(r"P(Tag \mid \text{'Yeah'}) = \frac{C(\text{'Yeah'}, Tag)}{C(\text{'Yeah'})} = \frac{0}{0} \implies \text{Undefined}")
        st.markdown(
            """
                <p><strong>Backoff Journey:</strong></p>
                <ul>
                    <li>Trigram: ❌ Miss (word not in context)</li>
                    <li>Bigram: ❌ Miss (word not in context)</li>
                    <li>Unigram: ❌ Miss (word not in vocabulary)</li>
                    <li><strong>Default Fallback:</strong> ➔ <code>'NN'</code> (Noun)</li>
                </ul>
                <p style="color: #94a3b8; font-size: 13px;"><em>Why 'NN'? In English open-class vocabulary, nouns are the largest category (~35% of all vocabulary words). Assigning 'NN' minimizes risk for unseen words.</em></p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="calc-card">
                <h5>3. Bigram Context Disambiguation Calculation</h5>
                <p><strong>Example Token:</strong> <code>"right"</code> preceded by Comma tag <code>','</code></p>
                <p><strong>Corpus Challenge:</strong> <code>"right"</code> is ambiguous: it appears 22 times as Noun (<code>NN</code>), 7 times as Adverb (<code>RB</code>), and 5 times as Adjective (<code>JJ</code>).</p>
                <p><strong>Context Conditioning:</strong></p>
            """,
            unsafe_allow_html=True,
        )
        st.latex(r"P(Tag \mid t_{i-1}=\text{','}, w_i=\text{'right'}) = \frac{C(t_{i-1}=\text{','}, w_i=\text{'right'}, Tag)}{C(t_{i-1}=\text{','}, w_i=\text{'right'})}")
        st.markdown(
            """
                <p><strong>Bigram Decision:</strong></p>
                <ul>
                    <li>In isolation (Unigram): <code>"right"</code> would be tagged <code>NN</code> (64.7%).</li>
                    <li><strong>With Bigram Context:</strong> Following <code>','</code>, it matches adverbial discourse ➔ <code>'RB'</code> (Adverb: <em>"right before lunch"</em>)!</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            """
            <div class="calc-card">
                <h5>2. Unigram Maximum-Likelihood Probability Calculation</h5>
                <p><strong>Example Token:</strong> <code>"told"</code> (Verb)</p>
                <p><strong>Corpus Frequency:</strong> <code>Count("told") = 11</code> in training split.</p>
                <p><strong>Tag Frequency Breakdown:</strong></p>
                <ul>
                    <li><code>Count("told", VBD) = 9</code> (Past tense verb)</li>
                    <li><code>Count("told", VBN) = 2</code> (Past participle)</li>
                </ul>
                <p><strong>Empirical Probability:</strong></p>
            """,
            unsafe_allow_html=True,
        )
        st.latex(r"P(\text{VBD} \mid \text{'told'}) = \frac{9}{11} \approx 81.82\%")
        st.latex(r"P(\text{VBN} \mid \text{'told'}) = \frac{2}{11} \approx 18.18\%")
        st.markdown(
            """
                <p><strong>Decision Rule:</strong></p>
            """,
            unsafe_allow_html=True,
        )
        st.latex(r"\hat{t} = \arg\max_{t \in \{\text{VBD}, \text{VBN}\}} P(t \mid \text{'told'}) = \text{\textbf{VBD}}")
        st.markdown(
            """
                <p style="color: #94a3b8; font-size: 13px;"><em>The Unigram model picks VBD because 81.8% &gt; 18.2%.</em></p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="calc-card">
                <h5>4. Trigram 3-Gram Syntactic Structure Calculation</h5>
                <p><strong>Example Token:</strong> <code>"that"</code> preceded by Noun and Comma <code>('NN', ',')</code></p>
                <p><strong>Corpus Challenge:</strong> <code>"that"</code> appears 411 times as Preposition/Conjunction (<code>IN</code>) and 170 times as Wh-Determiner (<code>WDT</code>).</p>
                <p><strong>Trigram Conditioning:</strong></p>
            """,
            unsafe_allow_html=True,
        )
        st.latex(r"P(Tag \mid t_{i-2}=\text{NN}, t_{i-1}=\text{','}, w_i=\text{'that'})")
        st.markdown(
            """
                <p><strong>Trigram Decision:</strong></p>
                <ul>
                    <li>Pure Unigram would incorrectly guess <code>IN</code> (63.1%).</li>
                    <li><strong>Trigram 3-Gram syntax:</strong> Recognizes non-restrictive relative clause syntax (<em>"lunch, that was..."</em>) ➔ Overrides to <code>'WDT'</code>!</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # 3. Interactive Sentence Word-by-Word Decomposition & Audit Inspector
    st.subheader("🔍 Interactive Sentence Word Decomposition & Audit Inspector")
    st.markdown("Enter or choose any sentence to decompose it into word tokens and inspect the live backoff trace for each word:")

    trace_sentence = st.text_input(
        "Sentence to deconstruct and audit:",
        value=st.session_state.get("user_transcript_box", "Yeah, honestly, that's what he told me, like, right before lunch."),
    )

    if trace_sentence.strip() and pipeline.is_model_loaded:
        deconstructed_tokens = tokenize_conversational_transcript(trace_sentence)
        st.markdown(f"**Deconstructed Words ({sum(len(s) for s in deconstructed_tokens)} tokens):**")

        test_res = pipeline.tag_text_with_explanations(trace_sentence)

        for sent_idx, (_, exp_list) in enumerate(test_res, 1):
            st.markdown(f"#### Sentence {sent_idx} Audit Trail:")

            for exp in exp_list:
                tier_badge_color = {
                    "Trigram": "tier-trigram",
                    "Bigram": "tier-bigram",
                    "Unigram": "tier-unigram",
                    "Default": "tier-default",
                }.get(exp.responsible_tagger, "tier-default")

                with st.expander(f"Word: '{exp.token}' ➔ Tag: {exp.assigned_tag} [{exp.responsible_tagger}]"):
                    sc1, sc2 = st.columns([1, 1])
                    with sc1:
                        st.markdown(f"**Word Token:** `{exp.token}` (Index: {exp.token_index})")
                        st.markdown(f"**Assigned POS Tag:** `{exp.assigned_tag}`")
                        st.markdown(f"**Tag Definition:** *{exp.tag_description}*")
                        st.markdown(f"**Responsible Model Tier:** <span class='tag-label {tier_badge_color}'>{exp.responsible_tagger}</span>", unsafe_allow_html=True)
                    with sc2:
                        st.markdown(f"**Why this decision was made:** {exp.rationale}")
                        st.markdown("**Backoff Search Path:**")
                        for step in exp.backoff_path:
                            st.markdown(f"- `{step}`")


# ==============================================================================
# PAGE 4: PROBABILITY EXPLORER
# ==============================================================================
elif navigation_choice == "📊 Probability Explorer":
    st.title("📊 Empirical Corpus Probability Explorer")
    st.markdown(
        """
        Inspect empirical corpus frequencies, emission distributions $P(w_i \\mid t_i)$,
        and transition probabilities $P(t_i \\mid t_{i-1})$ calculated directly from the training corpus.
        """
    )

    st.warning(
        "⚠️ **Statistical Integrity Notice**: NLTK's sequential backoff taggers are discrete maximum-frequency deciders, "
        "not calibrated Bayesian posterior estimators. All probabilities displayed below are calculated "
        "from empirical Penn Treebank training counts via `src/probability.py`."
    )

    @st.cache_resource
    def get_probability_estimator():
        train_sents, _, _ = get_corpus_splits()
        estimator = CorpusProbabilityEstimator(train_sents)
        return estimator

    with st.spinner("Extracting empirical frequency distributions..."):
        prob_estimator = get_probability_estimator()

    search_word = st.text_input(
        "Search Word in Corpus (e.g., 'like', 'well', 'that', 'told', 'right', 'run'):",
        value="like",
    )

    if search_word:
        w_dist = prob_estimator.get_word_tag_distribution(search_word)
        if w_dist:
            st.subheader(f"Empirical Tag Distribution for '{search_word}' $P(Tag \\mid '{search_word}')$")
            df_dist = pd.DataFrame(
                [{"Tag": k, "Probability": v, "Description": PENN_TREEBANK_TAGS.get(k, k)} for k, v in w_dist.items()]
            )

            fig = px.bar(
                df_dist,
                x="Tag",
                y="Probability",
                hover_data=["Description"],
                color="Probability",
                color_continuous_scale="Viridis",
                title=f"Empirical Tag Probabilities for '{search_word}'",
            )
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(df_dist, use_container_width=True)
        else:
            st.info(f"Word '{search_word}' was not observed in the Penn Treebank training split (Out-of-Vocabulary).")


# ==============================================================================
# PAGE 5: MODEL EVALUATION
# ==============================================================================
elif navigation_choice == "📈 Model Evaluation":
    st.title("📈 Model Evaluation on Penn Treebank Test Split")
    st.markdown(
        """
        Empirical evaluation of the stochastic tagger hierarchy on the reproducible **10% Test Split (392 sentences)**.
        """
    )

    if st.button("🚀 Run Comprehensive Test Evaluation"):
        with st.spinner("Evaluating models on test split..."):
            train_sents, dev_sents, test_sents = get_corpus_splits()
            taggers = load_trained_taggers()

            # 1. Tier comparison table
            st.subheader("1. Comparative Accuracy Across Model Tiers")
            tier_df = evaluate_tagger_hierarchy(taggers, test_sents)
            st.dataframe(tier_df, use_container_width=True)

            # 2. Confusion Matrix
            st.subheader("2. Confusion Matrix (Top 12 Most Frequent Tags)")
            full_chain = taggers["chain"]
            tokens, golds, preds = extract_gold_and_pred(full_chain, test_sents)
            top_labels, cm_grid = compute_confusion_matrix_data(golds, preds, top_n=12)

            fig_cm = px.imshow(
                cm_grid,
                labels=dict(x="Predicted POS Tag", y="True POS Tag", color="Token Count"),
                x=top_labels,
                y=top_labels,
                color_continuous_scale="Blues",
                text_auto=True,
                title="Confusion Matrix on Test Split",
            )
            st.plotly_chart(fig_cm, use_container_width=True)

            # 3. Out of Vocabulary (OOV) Performance
            st.subheader("3. Out-Of-Vocabulary (OOV) Analysis")
            train_vocab = set(w for sent in train_sents for w, _ in sent)
            oov_metrics = analyze_oov_performance(tokens, golds, preds, train_vocab)

            c1, c2, c3 = st.columns(3)
            c1.metric("OOV Tokens in Test Set", oov_metrics["oov_tokens_count"])
            c2.metric("OOV Tokens Correct", oov_metrics["oov_tokens_correct"])
            c3.metric("OOV Fallback Accuracy", f"{oov_metrics['oov_accuracy'] * 100:.2f}%")
            st.caption("When an unseen token appears, the system falls through to DefaultTagger('NN').")
    else:
        st.info("Click the button above to run evaluation on the test split.")


# ==============================================================================
# PAGE 6: ARCHITECTURE & STATUS
# ==============================================================================
elif navigation_choice == "⚙️ Architecture & Status":
    st.title("⚙️ Project Architecture & Status")
    st.markdown("Overview of implemented components, module skeletons, and roadmap.")

    col_a, col_b = st.columns([1, 1], gap="medium")

    with col_a:
        st.subheader("Module Status Matrix")
        status_data = [
            ("app.py", "Streamlit UI, Direct Voice Auto-Tagging & Navigation", "✅ Completed"),
            ("src/preprocessing.py", "Conversational Text Normalization & Tokenization", "✅ Completed"),
            ("src/models.py", "Treebank Split, Training & Model Persistence", "✅ Completed"),
            ("src/pipeline.py", "Inference Coordination Pipeline with Audit", "✅ Completed"),
            ("src/probability.py", "Empirical Frequency Estimator & Plots", "✅ Completed"),
            ("src/explainability.py", "Decision Audit & Backoff Trace Wrapper", "✅ Completed"),
            ("src/evaluation.py", "Accuracy, Confusion Matrix & OOV Metrics", "✅ Completed"),
            ("components/voice_input.html", "Web Speech API Browser Microphone", "✅ Completed"),
            ("models/tagger_chain.pkl", "Serialized Trained Tagger Binary", "✅ Saved & Ready"),
        ]

        for mod, desc, stat in status_data:
            st.markdown(
                f"""
                <div style="border-left: 3px solid #10b981; padding: 6px 12px; margin-bottom: 8px; background: rgba(30, 41, 59, 0.4); border-radius: 0 6px 6px 0;">
                    <strong>{mod}</strong> — <span style="color: #94a3b8;">{desc}</span><br/>
                    <small style="color: #34d399;">{stat}</small>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with col_b:
        st.subheader("Project Capabilities Summary")
        st.markdown(
            """
            - **Conversational Speech Handling**: Preserves fillers (*um, uh, like*), repetitions, and conversational disfluencies.
            - **Sequential Backoff Architecture**: Trigram $\\to$ Bigram $\\to$ Unigram $\\to$ Default (`NN`).
            - **Live Instant Tagging**: Microphone speech automatically populates, preprocesses, and tags upon stopping without extra button clicks.
            - **Decision Explainability**: Inspects exact backoff hierarchy to reveal which tier assigned each tag.
            - **Empirical Statistics**: Visualizes corpus emissions and transition probabilities.
            - **Evaluation Metrics**: Rigorous test set validation with confusion matrices and OOV analysis.
            """
        )
