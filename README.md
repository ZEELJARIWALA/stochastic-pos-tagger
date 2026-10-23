# Stochastic POS Tagger Lab

> **AI/ML Mini Project**: Stochastic Part-of-Speech Tagging for Conversational Speech Transcripts using NLTK Penn Treebank sequential backoff models.

---

## 1. Project Overview

**Stochastic POS Tagger Lab** is an interactive web-based NLP laboratory built with Python and Streamlit. It investigates how stochastic n-gram sequence models tag spontaneous conversational speech transcripts captured via microphone or typed input.

### Key Conversational Features Preserved
- **Speech Fillers & Hesitations**: *um, uh, er, ah, hmm, you know, like, I mean*
- **Repeated Words & Self-Corrections**: *I went to the, to the store*
- **Informal Syntax & Dialogue Markers**: *yeah, well, sounds good*
- **Preserved Surface Forms**: No stop-word removal, stemming, or lemmatization, ensuring verbatim syntactic context is preserved for POS tagging.

---

## 2. Model Architecture & Backoff Hierarchy

The core tagging engine uses NLTK's sequential backoff architecture trained on the Penn Treebank corpus:

```text
[ Input Conversational Sentence ]
               │
               ▼
   [ Trigram Tagger: P(t_i | t_{i-2}, t_{i-1}) ]  ──(Context Miss)──►
               │                                                    │
         (Context Hit)                                              │
               ▼                                                    ▼
       Assign POS Tag                              [ Bigram Tagger: P(t_i | t_{i-1}) ]  ──(Context Miss)──►
                                                                   │                                      │
                                                             (Context Hit)                                │
                                                                   ▼                                      ▼
                                                           Assign POS Tag                 [ Unigram Tagger: P(t_i | w_i) ]  ──(OOV Miss)──►
                                                                                                         │                                │
                                                                                                   (Word Found)                           │
                                                                                                         ▼                                ▼
                                                                                                   Assign POS Tag               [ Default Tagger: "NN" ]
```

### Academic Disclaimers
1. **NLTK Backoff Taggers vs. Bayesian Posteriors**: NLTK's `TrigramTagger`, `BigramTagger`, and `UnigramTagger` are discrete, frequency-based deciders with deterministic fallback tiers. They do not output calibrated probabilities during inference.
2. **Empirical Statistics**: All transition and emission probabilities displayed in the *Probability Explorer* are calculated from empirical training-corpus frequency counts via `src/probability.py`.

---

## 3. Project Directory Structure

```text
stochastic-pos-tagger/
├── app.py                      # Main Streamlit web application & UI navigation
├── requirements.txt            # Python dependencies (Streamlit, NLTK, Pandas, Plotly, Scikit-learn)
├── README.md                   # Setup guide, architecture, and verification notes
├── .gitignore                  # Git ignore rules for caches, venvs, and model artifacts
├── src/
│   ├── __init__.py             # Package initialization
│   ├── preprocessing.py        # Conversational text cleaning and tokenization
│   ├── models.py               # Treebank split, cleaning (-NONE- removal), and tagger training
│   ├── pipeline.py             # Inference pipeline coordinating preprocessing and tagging
│   ├── probability.py          # Empirical frequency distributions and conditional probabilities
│   ├── explainability.py       # Decision audit trail and backoff path tracking
│   └── evaluation.py           # Metrics: Token accuracy, OOV accuracy, confusion matrices
├── components/
│   └── voice_input.html        # Browser microphone interface using Web Speech API
├── data/
│   ├── raw_transcripts/        # Conversational speech audio transcripts (.gitkeep)
│   └── annotated/              # Reference test annotations (.gitkeep)
├── models/                     # Serialized model artifacts (e.g., tagger_chain.pkl) (.gitkeep)
├── reports/                    # Generated confusion matrices, plots, and CSV reports (.gitkeep)
└── docs/                       # Academic notes, documentation, and diagrams (.gitkeep)
```

---

## 4. Setup and Installation Guide (Windows)

Follow these exact steps in **PowerShell** or **Command Prompt** to run the project locally.

### Step 1: Ensure Python 3.11+ is Installed
If Python is not already installed on your Windows machine, install it from [python.org](https://www.python.org/downloads/) (make sure to check **"Add python.exe to PATH"** during installation) or via the Microsoft Store.

Verify your installation:
```powershell
python --version
```

### Step 2: Navigate to Project Directory
```powershell
cd stochastic-pos-tagger
```

### Step 3: Create and Activate a Virtual Environment

**In PowerShell:**
```powershell
python -m venv .venv
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
.\.venv\Scripts\Activate.ps1
```

**In Command Prompt (cmd.exe):**
```cmd
python -m venv .venv
.\.venv\Scripts\activate.bat
```

### Step 4: Upgrade Pip and Install Dependencies
```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Step 5: Download Required NLTK Corpora
The Penn Treebank corpus is required for model training and evaluation:
```powershell
python -c "import nltk; nltk.download('treebank'); nltk.download('punkt'); nltk.download('punkt_tab')"
```

### Step 6: Launch the Streamlit Web Application
```powershell
streamlit run app.py
```
Streamlit will launch locally and open in your default browser at `http://localhost:8501`.

---

## 5. Browser Speech Recognition Compatibility

- **Web Speech API**: Handled in `components/voice_input.html`. Supported natively in Chromium-based browsers (**Google Chrome**, **Microsoft Edge**).
- **Fallback**: In browsers without native speech recognition support (such as Mozilla Firefox), the application alerts the user and provides a typed conversational transcript text area.

---

## 6. Implementation Status & Milestones

| Module / Component | Responsibility | Current Milestone Status |
| :--- | :--- | :--- |
| `app.py` | Streamlit Dashboard & Navigation | **Completed (Milestone 1)** |
| `components/voice_input.html` | Microphone Speech Recognition | **Completed (Milestone 1)** |
| `src/preprocessing.py` | Conversational Tokenizer & Fillers | **Completed (Milestone 1)** |
| `src/models.py` | Treebank Split & Model Training | **Skeleton & Specs Ready (Milestone 1)** |
| `src/pipeline.py` | Tagging Inference Coordination | **Skeleton & Specs Ready (Milestone 1)** |
| `src/probability.py` | Empirical Corpus Probabilities | **Skeleton & Specs Ready (Milestone 1)** |
| `src/explainability.py` | Backoff Decision Audit Trace | **Skeleton & Specs Ready (Milestone 1)** |
| `src/evaluation.py` | Accuracy, Confusion Matrix, OOV | **Skeleton & Specs Ready (Milestone 1)** |
| `models/tagger_chain.pkl` | Trained Model Binary Artifact | *Pending Milestone 2 Training* |

---

## 7. Verification and Testing Status

- **Static Code Analysis**: All modules (`preprocessing.py`, `models.py`, `pipeline.py`, `probability.py`, `explainability.py`, `evaluation.py`, `app.py`) have been verified for syntax correctness, internal consistency, relative paths, and clean import decoupling.
- **Untested Host Commands**: Because the host environment currently lacks an active Python executable in system `PATH`, commands involving `python -m venv`, `pip install -r requirements.txt`, and `streamlit run app.py` have not been executed on this host machine and must be run by the user once Python 3.11+ is activated.
