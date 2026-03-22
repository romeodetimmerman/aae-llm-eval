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


def save_sampled_csvs(sampled_data, output_dir):
    """
    write three csv exports: metadata, rq1 (context + target), rq2 (contexts split by turn)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    rq1_path = output_dir / "coraal_samples_RQ1.csv"
    rq2_path = output_dir / "coraal_samples_RQ2.csv"
    meta_path = output_dir / "coraal_samples_metadata.csv"

    # RQ1: context split by turn, also include features to annotate
    col_names_rq1 = [
        "segment_id",
        "turn_id",
        "turn_index",
        "context_turn",
        "habitual_be",
        "copula_deletion",
        "third_person_s_deletion",
        "possessive_s_deletion",
        "multiple_negation",
    ]

    # RQ2: context + target
    col_names_rq2 = [
        "sample_id",
        "context",
        "target",
    ]

    # keep metadata separate
    col_names_meta = [
        "sample_id",
        "source_file",
        "start_line",
        "context_word_count",
        "target_word_count",
        "contains_aint",
    ]

    with (
        rq1_path.open("w", encoding="utf8", newline="") as f_rq1, 
        rq2_path.open("w", encoding="utf8", newline="") as f_rq2,
        meta_path.open("w", encoding="utf8", newline="") as f_meta
    ):
        writer_rq1 = csv.DictWriter(f_rq1, fieldnames=col_names_rq1)
        writer_rq2 = csv.DictWriter(f_rq2, fieldnames=col_names_rq2)
        writer_meta = csv.DictWriter(f_meta, fieldnames=col_names_meta)
        writer_rq1.writeheader()
        writer_rq2.writeheader()
        writer_meta.writeheader()

        feature_dict = {
            "habitual_be": 0,
            "copula_deletion": 0,
            "third_person_s_deletion": 0,
            "possessive_s_deletion": 0,
            "multiple_negation": 0,
        }

        turn_id = 1
        for segment_id, row in enumerate(sampled_data):
            sample_id = segment_id + 1
            context_turns = row["context_turns"]
            target_turn = row["target_turn"]

            context_str = "\n".join(
                f"{speaker}\t{text}" for speaker, text in context_turns
            )
            target_str = f"{target_turn[0]}\t{target_turn[1]}"

            full_window = context_turns + [target_turn]
            contains_aint = window_contains_aint(full_window)

            writer_meta.writerow(
                {
                    "sample_id": sample_id,
                    "source_file": row["source_file"],
                    "start_line": row["start_line"],
                    "context_word_count": row["context_word_count"],
                    "target_word_count": row["target_word_count"],
                    "contains_aint": int(contains_aint),
                }
            )

            for turn_index, (speaker, text) in enumerate(context_turns, start=1):
                context_turn_line = f"{speaker}\t{text}"
                writer_rq1.writerow(
                    {
                        "segment_id": sample_id,
                        "turn_id": turn_id,
                        "turn_index": turn_index,
                        "context_turn": context_turn_line,
                        **feature_dict,
                    }
                )
                turn_id += 1
            
            writer_rq2.writerow(
                {
                    "sample_id": sample_id,
                    "context": context_str,
                    "target": target_str,
                }
            )


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

    # save sampled data to csv files
    save_sampled_csvs(sampled_data, SAMPLED_TRANSCRIPTS_DIR)
    print(f"successfully sampled {len(sampled_data)} segments and saved them to {SAMPLED_TRANSCRIPTS_DIR}")


if __name__ == "__main__":
    random.seed(SEED)
    sample_coraal()
