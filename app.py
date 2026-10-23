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

# Custom Styling for polished academic UI
st.markdown(
    """
    <style>
    .metric-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7), rgba(15, 23, 42, 0.9));
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 16px 20px;
        margin-bottom: 15px;
    }
    .status-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: 600;
        margin-right: 8px;
    }
    .status-ready {
        background: rgba(16, 185, 129, 0.2);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.4);
    }
    .status-wip {
        background: rgba(99, 102, 241, 0.2);
        color: #a5b4fc;
        border: 1px solid rgba(99, 102, 241, 0.4);
    }
    .tag-chip {
        display: inline-flex;
        align-items: center;
        background: #1e293b;
        border: 1px solid #475569;
        border-radius: 6px;
        padding: 4px 8px;
        margin: 4px;
        font-family: inherit;
        font-size: 14px;
    }
    .tag-label {
        margin-left: 6px;
        padding: 2px 6px;
        border-radius: 4px;
        font-size: 11px;
        font-weight: bold;
    }
    .tier-trigram { background: #3b82f6; color: white; }
    .tier-bigram { background: #8b5cf6; color: white; }
    .tier-unigram { background: #10b981; color: white; }
    .tier-default { background: #f59e0b; color: black; }
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
# PAGE 1: LIVE SPEECH & TEXT INPUT (DIRECT PREPROCESSING & POS TAGGING)
# ==============================================================================
if navigation_choice == "🎙️ Live Speech & Text Input":
    st.title("🎙️ Conversational Speech & Real-Time POS Tagging")
    st.markdown(
        """
        Speak naturally through your microphone or type a conversational transcript.
        When you stop speaking, text is **automatically captured, preprocessed, and tagged**
        with Penn Treebank sequential backoff stochastic models.
        """
    )

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.subheader("Microphone Audio Capture")
        st.caption("Web Speech API client-side recognition with direct auto-trigger.")

        # Embed voice input HTML component if present
        if VOICE_HTML_PATH.exists():
            with open(VOICE_HTML_PATH, "r", encoding="utf-8") as f:
                voice_html_content = f.read()
            st.components.v1.html(voice_html_content, height=190, scrolling=False)
        else:
            st.warning("Voice component template `components/voice_input.html` not found.")

        st.info(
            "💡 **Microphone Tip**: Click **Start Speaking**, allow microphone access, speak your sentence, "
            "then click **Stop Recording**. The transcript directly populates and tags below!"
        )

    with col2:
        st.subheader("Conversational Transcript")
        st.caption("Preserves fillers (um, uh, like), repetitions, and conversational disfluencies.")

        # Read speech parameter from URL if microphone was used
        spoken_param = st.query_params.get("spoken", None)
        timestamp_param = st.query_params.get("t", None)

        if spoken_param and timestamp_param != st.session_state.get("last_voice_t"):
            st.session_state["last_voice_t"] = timestamp_param
            st.session_state["user_transcript_val"] = spoken_param

        sample_presets = {
            "Select sample preset...": "",
            "Dialogue with Fillers": "Yeah, honestly, that's what he told me, like, right before lunch.",
            "Hesitations & Self-Corrections": "Um, uh, I think, you know, we should go to the, to the main office.",
            "Informal Conversational": "Well, hmm, uh-huh, let me see if the data is ready.",
            "Discourse Markers": "I mean, like, he was actually right about that.",
        }

        selected_preset = st.selectbox("Load Sample Preset:", list(sample_presets.keys()))

        if selected_preset != "Select sample preset...":
            current_text = sample_presets[selected_preset]
        else:
            current_text = st.session_state.get(
                "user_transcript_val",
                "Yeah, honestly, that's what he told me, like, right before lunch."
            )

        user_text = st.text_area(
            "Conversational Transcript Text Area:",
            value=current_text,
            height=110,
            placeholder="Speak via microphone or type conversational speech here...",
        )
        st.session_state["user_transcript_val"] = user_text

    st.markdown("---")

    # DIRECT & AUTOMATIC PREPROCESSING & POS TAGGING (Runs immediately if user_text is present!)
    if user_text.strip():
        cleaned = clean_conversational_text(user_text)
        sentences = tokenize_conversational_transcript(user_text)

        # 1. PREPROCESSING OUTPUT SECTION (Exactly as requested)
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
        [{"Tag": k, "Description": v} for k, v in list(PENN_TREEBANK_TAGS.items())[:15]]
    )
    st.table(ref_df)


# ==============================================================================
# PAGE 3: ALGORITHM EXPLORER
# ==============================================================================
elif navigation_choice == "🔍 Algorithm & Backoff Trace":
    st.title("🔍 Algorithm & Sequential Backoff Trace")
    st.markdown(
        """
        Examine how sequential stochastic taggers make decisions and fall back when higher-order n-gram context is missing.
        """
    )

    st.markdown(
        """
        <div class="metric-card">
            <h4>Hierarchy of N-gram Context</h4>
            <table style="width: 100%; font-size: 14px; border-collapse: collapse;">
                <tr style="border-bottom: 1px solid #334155;">
                    <th style="padding: 8px; text-align: left;">Model Tier</th>
                    <th style="padding: 8px; text-align: left;">Conditioning Context</th>
                    <th style="padding: 8px; text-align: left;">Decision Rule</th>
                </tr>
                <tr style="border-bottom: 1px solid #1e293b;">
                    <td style="padding: 8px;"><strong style="color: #3b82f6;">Trigram</strong></td>
                    <td style="padding: 8px;"><code>P(t_i | t_{i-2}, t_{i-1}, w_i)</code></td>
                    <td style="padding: 8px;">Checks if preceding 2 tags and current word occurred in training.</td>
                </tr>
                <tr style="border-bottom: 1px solid #1e293b;">
                    <td style="padding: 8px;"><strong style="color: #8b5cf6;">Bigram</strong></td>
                    <td style="padding: 8px;"><code>P(t_i | t_{i-1}, w_i)</code></td>
                    <td style="padding: 8px;">Falls back to 1 preceding tag if trigram context was never seen.</td>
                </tr>
                <tr style="border-bottom: 1px solid #1e293b;">
                    <td style="padding: 8px;"><strong style="color: #10b981;">Unigram</strong></td>
                    <td style="padding: 8px;"><code>P(t_i | w_i)</code></td>
                    <td style="padding: 8px;">Assigns the token's most frequent tag in the corpus regardless of context.</td>
                </tr>
                <tr>
                    <td style="padding: 8px;"><strong style="color: #f59e0b;">Default</strong></td>
                    <td style="padding: 8px;">None (<code>"NN"</code>)</td>
                    <td style="padding: 8px;">Robust fallback for unseen out-of-vocabulary (OOV) tokens.</td>
                </tr>
            </table>
        </div>
        """,
        unsafe_allow_html=True,
    )

    sample_test = st.selectbox(
        "Select conversational sample to inspect backoff path:",
        [
            "Yeah, honestly, that's what he told me, like, right before lunch.",
            "Um, the new director was announced today.",
            "Well, I went to the store and bought three books.",
        ],
    )

    if pipeline.is_model_loaded:
        test_res = pipeline.tag_text_with_explanations(sample_test)
        for sent_idx, (_, exp_list) in enumerate(test_res, 1):
            st.markdown(f"**Sentence {sent_idx} Audit Trail:**")
            for exp in exp_list:
                with st.expander(f"Token: `{exp.token}` ➔ Assigned Tag: **`{exp.assigned_tag}`** [{exp.responsible_tagger}]"):
                    st.write(f"**Tag Description:** {exp.tag_description}")
                    st.write(f"**Responsible Tier:** {exp.responsible_tagger}")
                    st.write(f"**Rationale:** {exp.rationale}")
                    st.write("**Backoff Search Path:**")
                    for step in exp.backoff_path:
                        st.write(f"- `{step}`")


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
        "Search Word in Corpus (e.g., 'like', 'well', 'that', 'before', 'run'):",
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
