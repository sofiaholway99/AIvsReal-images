"""
AI vs Real Image Judgment Task — Streamlit App
------------------------------------------------
What this does
- Presents 7 (or more) trials of image pairs (AI vs Real).
- Randomizes left/right position per trial and trial order per participant.
- Records response, correctness, reaction time, and score.
- Saves per-participant results to /results as CSV and appends a summary row to results/all_results.csv.

How to run
1) Install deps:  
   pip install streamlit pandas numpy pillow

2) Put a CSV named trials.csv next to this file with columns:
   pair_id,real_url,ai_url
   (You can also point to local file paths instead of URLs.)

   Example trials.csv:
   pair_id,real_url,ai_url
   p1,https://example.com/real1.jpg,https://example.com/ai1.jpg
   p2,https://example.com/real2.jpg,https://example.com/ai2.jpg
   p3,https://example.com/real3.jpg,https://example.com/ai3.jpg
   p4,https://example.com/real4.jpg,https://example.com/ai4.jpg
   p5,https://example.com/real5.jpg,https://example.com/ai5.jpg
   p6,https://example.com/real6.jpg,https://example.com/ai6.jpg
   p7,https://example.com/real7.jpg,https://example.com/ai7.jpg

3) Run the app:  
   streamlit run app.py

4) Share it: participants open the URL you get in the terminal (e.g., http://localhost:8501 or your server address).

Notes
- If you need GDPR/consent, add a checkbox before starting and avoid collecting personal identifiers.
- For online sharing with public links: deploy on Streamlit Community Cloud or run on a server.
- For larger studies, consider jsPsych (browser-only) or PsychoPy/PsychoJS.
"""

import time
import uuid
from pathlib import Path
import random

import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st

# -----------------------------
# Utilities
# -----------------------------

def load_image(path_or_url: str) -> Image.Image:
    """Lazy load with Pillow. Streamlit can also display URLs directly,
    but we use PIL to catch errors more gracefully.
    """
    try:
        return Image.open(path_or_url)
    except Exception:
        # Fallback: let Streamlit try to display the URL directly
        return None


def ensure_dirs():
    Path("results").mkdir(exist_ok=True)


def read_trials(csv_path: str = "trials.csv") -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    req = {"pair_id", "real_url", "ai_url"}
    missing = req - set(df.columns)
    if missing:
        raise ValueError(f"trials.csv missing columns: {missing}")
    return df.copy()

def build_trials_df(trials_src_df: pd.DataFrame) -> pd.DataFrame:
    """Return a randomized trials dataframe with per-pair left/right shuffle and randomized order."""
    rows = []
    rng = random.SystemRandom()  # better randomness than default
    for _, row in trials_src_df.iterrows():
        left_is_real = bool(rng.getrandbits(1))
        rows.append({
            "pair_id": row["pair_id"],
            "left_url": row["real_url"] if left_is_real else row["ai_url"],
            "right_url": row["ai_url"] if left_is_real else row["real_url"],
            "correct_side": "left" if left_is_real else "right",
        })
    rng.shuffle(rows)  # randomize trial order
    return pd.DataFrame(rows)

# -----------------------------
# App State Management
# -----------------------------

def init_state():
    if "initialized" not in st.session_state:
        st.session_state.initialized = True
        st.session_state.participant_id = ""
        st.session_state.trials = None
        st.session_state.order = []
        st.session_state.i = 0
        st.session_state.records = []
        st.session_state.start_time = None
        st.session_state.consent = False


# -----------------------------
# UI Components
# -----------------------------

def header():
    st.title("AI vs Real — Image Judgment Test")
    st.write(
        "Identify which photo is REAL. You'll see pairs of images. Click the real one."
    )


@st.cache_data
def load_trials_cached() -> pd.DataFrame:
    return read_trials("trials.csv")


def start_block():
    with st.form("start_form"):
        st.session_state.participant_id = st.text_input(
            "Participant ID (anonymous code is OK)",
            value=str(uuid.uuid4())[:8],
            help="Use a code, not a name, unless you have explicit consent.",
        )
        st.session_state.consent = st.checkbox(
            "I consent to take part in this study and for my anonymous responses to be stored.",
            value=True,
        )
        submitted = st.form_submit_button("Start test ▶")

    if submitted:
        if not st.session_state.consent:
            st.warning("You must consent to proceed.")
            return False

        try:
            trials_src = load_trials_cached()  # reads trials.csv
        except Exception as e:
            st.error(f"Error loading trials.csv: {e}")
            return False

        # >>> NEW: build a randomized set of trials for this participant
        st.session_state.trials = build_trials_df(trials_src)
        st.session_state.order = list(range(len(st.session_state.trials)))
        st.session_state.i = 0
        st.session_state.records = []
        st.session_state.start_time = time.time()
        st.success("Loaded! Starting now…")
        st.rerun()

    # falls through to the end of start_block()
    return False



def show_trial():
    i = st.session_state.i
    trials = st.session_state.trials
    trial = trials.iloc[i]

    st.subheader(f"Trial {i + 1} of {len(trials)}")
    c1, c2 = st.columns(2, gap="large")

    # We try to load with PIL for better sizing; if None, let st.image try URL directly
    left_img = load_image(trial.left_url)
    right_img = load_image(trial.right_url)

    with c1:
        st.write("**Left**")
        if left_img is not None:
            st.image(left_img, use_container_width=True)
        else:
            st.image(trial.left_url, use_container_width=True)
        left_clicked = st.button("This is REAL", key=f"left_{i}")

    with c2:
        st.write("**Right**")
        if right_img is not None:
            st.image(right_img, use_container_width=True)
        else:
            st.image(trial.right_url, use_container_width=True)
        right_clicked = st.button("This is REAL", key=f"right_{i}")

    # Record response
    if left_clicked or right_clicked:
        rt = time.time() - st.session_state.start_time
        choice = "left" if left_clicked else "right"
        correct = int(choice == trial.correct_side)

        st.session_state.records.append(
            {
                "participant_id": st.session_state.participant_id,
                "trial_index": i,
                "pair_id": trial.pair_id,
                "left_url": trial.left_url,
                "right_url": trial.right_url,
                "correct_side": trial.correct_side,
                "choice": choice,
                "correct": correct,
                "rt_sec": round(rt, 3),
                "timestamp": pd.Timestamp.now().isoformat(),
            }
        )
        st.session_state.i += 1
        st.session_state.start_time = time.time()
        st.rerun()


def finish_block():
    # Compute score
    df = pd.DataFrame(st.session_state.records)
    accuracy = df["correct"].mean() if not df.empty else 0.0
    st.success(f"Done! Accuracy: {accuracy * 100:.1f}%")

    # Per-pair accuracy
    st.write("### Per-pair accuracy")
    pair_acc = df.groupby("pair_id")["correct"].mean().reset_index()
    st.dataframe(pair_acc)

    # Save results
    ensure_dirs()
    participant = st.session_state.participant_id
    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")

    per_participant_path = Path("results") / f"{participant}_{ts}.csv"
    df.to_csv(per_participant_path, index=False)

    # Append to global summary
    summary_path = Path("results") / "all_results.csv"
    summary_row = pd.DataFrame(
        {
            "participant_id": [participant],
            "timestamp": [pd.Timestamp.now().isoformat()],
            "n_trials": [len(df)],
            "accuracy": [accuracy],
            "median_rt_sec": [float(df.rt_sec.median()) if not df.empty else np.nan],
        }
    )
    if summary_path.exists():
        all_res = pd.read_csv(summary_path)
        all_res = pd.concat([all_res, summary_row], ignore_index=True)
        all_res.to_csv(summary_path, index=False)
    else:
        summary_row.to_csv(summary_path, index=False)

    st.info(
        f"Results saved to **{per_participant_path}** and appended to **{summary_path}**."
    )

    # >>> UPDATED: restart builds a brand-new randomized block
    if st.button("Restart for a new participant"):
        trials_src = load_trials_cached()
        st.session_state.trials = build_trials_df(trials_src)  # NEW randomization
        st.session_state.participant_id = str(uuid.uuid4())[:8]
        st.session_state.i = 0
        st.session_state.records = []
        st.session_state.start_time = time.time()
        st.rerun()


# -----------------------------
# Main
# -----------------------------

def main():
    st.set_page_config(page_title="AI vs Real Image Test", layout="wide")
    init_state()
    header()

    # If not started, show start form
    if st.session_state.trials is None:
        started = start_block()
        if not started and st.session_state.trials is None:
            st.stop()

    # If mid-task, show a trial
    if st.session_state.i < len(st.session_state.trials):
        show_trial()
        st.stop()

    # Otherwise, finished
    finish_block()


if __name__ == "__main__":
    main()
