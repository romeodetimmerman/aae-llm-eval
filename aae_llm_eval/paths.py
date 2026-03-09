from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"

CORAAL_TRANSCRIPTS_DIR = RAW_DATA_DIR / "coraal_transcripts"
CORAAL_METADATA_DIR = RAW_DATA_DIR / "coraal_metadata"

PREPROCESSED_TRANSCRIPTS_DIR = INTERIM_DATA_DIR / "preprocessed_transcripts"
CORAAL_METADATA_OUTPUT_PATH = INTERIM_DATA_DIR / "coraal_metadata.csv"
