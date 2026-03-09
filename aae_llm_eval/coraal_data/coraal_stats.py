"""
module to compute basic statistics about the CORAAL data used for this project
"""

import pandas as pd
from aae_llm_eval.paths import CORAAL_METADATA_OUTPUT_PATH, PREPROCESSED_TRANSCRIPTS_DIR


def compute_coraal_stats():
    """
    computes number of interviews, turns, tokens, and unique tokens in the CORAAL data
    """
    meta = pd.read_csv(CORAAL_METADATA_OUTPUT_PATH)
    
    # unique interviews from metadata
    n_interviews = meta["coraal_file"].nunique()
    
    # iterate over preprocessed transcripts
    n_turns = 0
    token_count = 0
    vocab = set()
    
    for path in PREPROCESSED_TRANSCRIPTS_DIR.glob("*.txt"):
        with path.open(encoding="utf8") as f:
            for line in f:
                n_turns += 1
                # speaker \t text
                parts = line.rstrip("\n").split("\t", maxsplit=1)
                if len(parts) == 2:
                    text = parts[1]
                    tokens = text.split()
                    token_count += len(tokens)
                    vocab.update(tokens)
    
    print(f"unique interviews: {n_interviews}")
    print(f"total turns: {n_turns}")
    print(f"total tokens: {token_count}")
    print(f"unique tokens: {len(vocab)}")


if __name__ == "__main__":
    compute_coraal_stats()
