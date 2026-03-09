"""
module to preprocesses the CORAAL data
- we filter out para/non-linguistic utterances (e.g. laughter, coughing, etc.)
- we merge consecutive speaker utterances into a single turn
- we only keep speaker labels and spoken turns

the module is partially adapted from Tessa Masis' CORAAL preprocessing script
- original paper: Masis, T., Neal, A., Green, L., & O’Connor, B. (2022, October). Corpus-guided contrast sets for morphosyntactic feature detection in low-resource English varieties. In Proceedings of the first workshop on NLP applications to field linguistics (pp. 11-25).
- original script: https://github.com/slanglab/CGEdit/blob/main/code/preprocessCORAAL.py
"""

import re
import pandas as pd
from aae_llm_eval.paths import CORAAL_TRANSCRIPTS_DIR, PREPROCESSED_TRANSCRIPTS_DIR, CORAAL_METADATA_OUTPUT_PATH


def preprocess_transcript(path):
    """
    preprocess a single transcript file
    """
    # regex patterns to remove para/non-linguistic utterances
    to_remove = [
        r"\([^)]*\)",
        r"<[^>]*>",
        r"/unintelligible/",
        r"/inaudible/",
        r"/[?]*/",
        r"[\[\]/]",
    ]

    # iterate over lines in the transcript file
    first = True
    current_speaker = None
    current_text = ""
    turns: list[tuple[str, str]] = []

    with path.open(encoding="utf8", errors="ignore") as f:
        for line in f:
            if first:
                first = False
                continue

            parts = line.split("\t")
            if len(parts) < 4:
                continue

            speaker = parts[1].strip()
            sent = parts[3]

            for pattern in to_remove:
                sent = re.sub(pattern, "", sent)

            sent = sent.strip()
            if not sent:
                continue

            if speaker == current_speaker:
                current_text += " " + sent
            else:
                if current_speaker is not None and current_text.strip():
                    turns.append((current_speaker, current_text.strip()))
                current_speaker = speaker
                current_text = sent

    if current_speaker is not None and current_text.strip():
        turns.append((current_speaker, current_text.strip()))

    # save the preprocessed transcript to a file
    PREPROCESSED_TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PREPROCESSED_TRANSCRIPTS_DIR / f"{path.stem}.txt"
    with out_path.open("w", encoding="utf8") as w:
        for speaker, text in turns:
            w.write(f"{speaker}\t{text}\n")


def preprocess_coraal():
    """
    preprocess a batch of transcript files that are present in the CORAAL metadata file
    """
    metadata = pd.read_csv(CORAAL_METADATA_OUTPUT_PATH)
    files_of_interest = set(metadata["coraal_file"].tolist())
    transcript_files = [file for file in CORAAL_TRANSCRIPTS_DIR.glob("**/*.txt") if file.name.replace(".txt", "") in files_of_interest]

    if not transcript_files:
        print("no matching transcript files found")
        return

    for file in transcript_files:
        preprocess_transcript(file)

    print(f"preprocessed {len(transcript_files)} transcript files")


if __name__ == "__main__":
    preprocess_coraal()
