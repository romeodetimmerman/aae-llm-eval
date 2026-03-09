"""
module to merge CORAAL metadata files and filter for specific interviewer demographics.
"""

import pandas as pd
from aae_llm_eval.paths import CORAAL_METADATA_DIR, CORAAL_METADATA_OUTPUT_PATH


def merge_metadata():
    """
    reads all CORAAL txt metadata, merges on common columns, 
    and filters for African American interviewers with close relationships
    """
    dataframes = []
    
    # get all metadata files
    files = sorted(CORAAL_METADATA_DIR.glob("*.txt"))
    
    if not files:
        print("no metadata files found.")
        return

    for filepath in files:
        df = pd.read_csv(filepath, sep="\t")
        dataframes.append(df)
        print(f"{filepath.name}: {len(df.columns)} columns")
        
    # find common columns
    common_columns = set.intersection(*(set(df.columns) for df in dataframes))
    common_columns = sorted(list(common_columns))
    print(f"found {len(common_columns)} common columns")

    # keep only common columns
    full_df = pd.concat([df[common_columns] for df in dataframes], ignore_index=True)

    # filter for african american interviewers and close relationships
    is_aa = full_df["Interviewer.Ethnicity"] == "African American"
    is_close = full_df["Interviewer.Relationship"].isin(["Friend", "Close Relationship"])
    
    filtered_df = full_df[is_aa & is_close].copy()

    # column selection
    cols_to_select = [
        "Age", "Audio.Folder", "CORAAL.File", "CORAAL.Spkr", 
        "CORAAL.Sub", "Edu.Group", "Education", "Gender", 
        "Interviewer.Age", "Interviewer.Code", "Interviewer.Ethnicity", 
        "Interviewer.Relationship", "Year.of.Interview"
    ]
    filtered_df = filtered_df[cols_to_select]

    # convert column names to snake case
    filtered_df.columns = filtered_df.columns.str.lower().str.replace(".", "_")

    # save output
    CORAAL_METADATA_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    filtered_df.to_csv(CORAAL_METADATA_OUTPUT_PATH, index=False)
    print(f"saved metadata for {len(filtered_df)} interviews to {CORAAL_METADATA_OUTPUT_PATH}")


if __name__ == "__main__":
    merge_metadata()
