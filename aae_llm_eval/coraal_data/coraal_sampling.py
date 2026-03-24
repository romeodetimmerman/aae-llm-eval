"""
module to randomly sample a subset of conversational segments from the CORAAL data
each sample includes 5 turns of context (RQ1) and 1 target turn (RQ2)
"""

import csv
import random
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


def target_not_in_context(context_turns, target_turn):
    """
    check if target text is absent from context turns
    """
    context_texts = {text.strip().lower() for _, text in context_turns}
    target_text = target_turn[1].strip().lower()
    return target_text not in context_texts


def collect_valid_windows(transcripts, n_context_turns):
    """
    collect valid windows (n_context_turns context + 1 target)
    """
    # init windows
    windows = []
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

            # skip windows where target text repeats in context
            if not target_not_in_context(context_turns, target_turn):
                continue

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

            # keep all valid windows
            windows.append(window_data)

    return windows


def window_turn_ids(window, n_context_turns):
    """
    return ids for all turns used by this window
    """
    base = window["start_line"]
    src = window["source_file"]
    window_size = n_context_turns + 1
    return {(src, base + k) for k in range(window_size)}


def sample_random_non_overlapping_windows(
    windows, n_samples, n_context_turns
):
    """
    sample windows without reusing any turns
    """
    candidates = list(windows)
    random.shuffle(candidates)

    used_turns = set()
    selected = []

    for window in candidates:
        # make sure windows do not share any turns
        turn_ids = window_turn_ids(window, n_context_turns)
        if not used_turns.isdisjoint(turn_ids):
            continue

        # keep track of used turns
        used_turns.update(turn_ids)
        selected.append(window)

        # stop as soon as we have enough samples
        if len(selected) >= n_samples:
            break

    return selected


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
        "is_was_generalization",
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
            "is_was_generalization": 0,
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

            writer_meta.writerow(
                {
                    "sample_id": sample_id,
                    "source_file": row["source_file"],
                    "start_line": row["start_line"],
                    "context_word_count": row["context_word_count"],
                    "target_word_count": row["target_word_count"],
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
    windows = collect_valid_windows(
        transcripts=transcripts, n_context_turns=N_CONTEXT_TURNS
    )

    total_valid_windows = len(windows)
    print(f"found {total_valid_windows} valid windows")

    # check if we have enough valid windows to sample from
    if total_valid_windows < N_SAMPLES:
        print(f"warning: only found {total_valid_windows} valid windows, sampling all")

    n_to_sample = min(N_SAMPLES, total_valid_windows)
    sampled_data = sample_random_non_overlapping_windows(
        windows=windows,
        n_samples=n_to_sample,
        n_context_turns=N_CONTEXT_TURNS,
    )

    # save sampled data to csv files
    save_sampled_csvs(sampled_data, SAMPLED_TRANSCRIPTS_DIR)
    print(f"successfully sampled {len(sampled_data)} segments and saved them to {SAMPLED_TRANSCRIPTS_DIR}")


if __name__ == "__main__":
    random.seed(SEED)
    sample_coraal()
