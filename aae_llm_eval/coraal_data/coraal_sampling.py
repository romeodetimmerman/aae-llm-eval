"""
module to randomly sample a subset of conversational 5-turn segments from the CORAAL data
and keep a single following turn for each sampled segment for RQ2
"""

import csv
import random
import re
from aae_llm_eval.paths import PREPROCESSED_TRANSCRIPTS_DIR, SAMPLED_TRANSCRIPTS_DIR

SEED = 42
N_SAMPLES = 100
N_CONTEXT_TURNS = 5
MIN_TURN_WORD_COUNT = 5  # filter out backchannel turns (e.g. "mhm", "yes", "ok", etc.)


def window_is_valid(window):
    """
    check if each turn contains at least MIN_TURN_WORD_COUNT words
    """
    window_turn_word_counts = [len(turn[1].split()) for turn in window]
    return min(window_turn_word_counts) >= MIN_TURN_WORD_COUNT


def window_contains_aint(window):
    """
    check if lexical item "ain't" occurs in window
    """
    for _, text in window:
        if re.search(r"\bain't\b", text, flags=re.IGNORECASE):
            return True
    return False


def collect_valid_windows(transcripts, n_context_turns):
    """
    collect valid windows and split aint presence
    """
    windows_with_aint = []
    windows_without_aint = []

    window_size = n_context_turns + 1

    # loop over all transcripts
    for path in transcripts:
        with path.open("r", encoding="utf8") as f:
            # each line is formatted as speaker \t text
            turns = []
            for line in f:
                line = line.strip()
                if "\t" not in line:
                    continue
                speaker, text = line.split("\t", 1)
                speaker = speaker.strip()
                text = text.strip()
                if not speaker or not text:
                    continue
                turns.append([speaker, text])

        # loop over all possible windows in the transcript
        for i in range(len(turns) - window_size + 1):
            window = turns[i : i + window_size]

            # check if window is valid
            if not window_is_valid(window):
                continue

            # get context and target turns
            context_turns = window[:n_context_turns]
            target_turn = window[n_context_turns]

            # get context and target text
            context_text = " ".join([turn[1] for turn in context_turns])
            target_text = target_turn[1]

            # get context and target word counts
            context_word_count = len(context_text.split())
            target_word_count = len(target_text.split())

            window_data = {
                "source_file": path.stem,
                "start_line": i,
                "context_word_count": context_word_count,
                "target_word_count": target_word_count,
                "context_turns": context_turns,
                "target_turn": target_turn,
            }

            # check if window contains aint
            if window_contains_aint(window):
                windows_with_aint.append(window_data)
            else:
                windows_without_aint.append(window_data)

    return windows_with_aint, windows_without_aint


def sample_balanced(windows_with_aint, windows_without_aint, n_samples):
    """
    sample half of segments from windows with and without "ain't"
    """
    # even sample size for simplicity
    assert n_samples % 2 == 0, "n_samples must be even"

    total_valid_windows = len(windows_with_aint) + len(windows_without_aint)

    # check if there are enough valid windows to sample from
    if total_valid_windows <= n_samples:
        return windows_with_aint + windows_without_aint

    # sample half of segments from each group
    half = n_samples // 2
    target_with_aint = min(half, len(windows_with_aint))
    target_without_aint = min(half, len(windows_without_aint))

    sampled_with_aint = random.sample(windows_with_aint, target_with_aint)
    sampled_without_aint = random.sample(windows_without_aint, target_without_aint)

    sampled_data = sampled_with_aint + sampled_without_aint

    # check if we sampled enough segments
    # if one of the two groups has less than half the samples, we need to sample from the remaining pool
    if len(sampled_data) < n_samples:
        remaining = n_samples - len(sampled_data)
        remaining_pool = [
            w
            for w in windows_with_aint + windows_without_aint
            if w not in sampled_data
        ]
        sampled_data.extend(random.sample(remaining_pool, remaining))

    # shuffle the sampled data
    random.shuffle(sampled_data)

    return sampled_data


def sample_coraal():
    """
    sample a subset of conversational segments from the CORAAL data
    each sample includes 5 turns of context (RQ1) and 1 target turn (RQ2)
    """
    transcripts = list(PREPROCESSED_TRANSCRIPTS_DIR.glob("*.txt"))

    print(f"searching for valid windows in {len(transcripts)} files")
    windows_with_aint, windows_without_aint = collect_valid_windows(
        transcripts=transcripts, n_context_turns=N_CONTEXT_TURNS
    )

    total_valid_windows = len(windows_with_aint) + len(windows_without_aint)
    print(f"found {len(windows_with_aint)} valid windows with \"ain't\" and {len(windows_without_aint)} without")

    # check if we have enough valid windows to sample from
    if total_valid_windows < N_SAMPLES:
        print(f"warning: only found {total_valid_windows} valid windows, sampling all")

    # sample half of segments from each group
    sampled_data = sample_balanced(
        windows_with_aint=windows_with_aint,
        windows_without_aint=windows_without_aint,
        n_samples=N_SAMPLES,
    )

    # save as a csv file
    SAMPLED_TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
    output_file = SAMPLED_TRANSCRIPTS_DIR / "coraal_samples.csv"

    # keep some metadata about the sampled segments
    fieldnames = [
        "source_file",
        "start_line",
        "context_word_count",
        "target_word_count",
        "context",
        "target",
    ]

    # write the sampled data to a csv file
    with output_file.open("w", encoding="utf8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in sampled_data:
            context_turns = row["context_turns"]
            target_turn = row["target_turn"]

            context_str = "\n".join(
                f"{speaker}\t{text}" for speaker, text in context_turns
            )
            target_str = f"{target_turn[0]}\t{target_turn[1]}"

            writer.writerow(
                {
                    "source_file": row["source_file"],
                    "start_line": row["start_line"],
                    "context_word_count": row["context_word_count"],
                    "target_word_count": row["target_word_count"],
                    "context": context_str,
                    "target": target_str,
                }
            )

    print(f"successfully sampled {len(sampled_data)} segments to {output_file}")


if __name__ == "__main__":
    # setting seed for reproducibility of the 100 samples
    random.seed(SEED)
    sample_coraal()
