
from pathlib import Path
import json
import time

import pandas as pd
import requests

# --------------------------------------------------
# 1. FILE LOCATIONS
# --------------------------------------------------

folder = Path(__file__).resolve().parent

template_file = folder / "ERa_BLA_Solubility_Lookup.csv"
progress_file = folder / "ERa_BLA_Solubility_Progress.csv"
final_file = folder / "ERa_BLA_Solubility_Results.csv"

# Start with a small test.
# Later, change 25 to None to process all remaining chemicals.
MAX_RECORDS = 25

if progress_file.exists():
    df = pd.read_csv(progress_file, dtype=str)
    print("Resuming previous solubility lookup.")
else:
    if not template_file.exists():
        raise FileNotFoundError(
            f"Cannot find the template:\n{template_file}"
        )

    df = pd.read_csv(template_file, dtype=str)
    print("Loaded solubility lookup template.")

if "PUBCHEM_CID" not in df.columns:
    raise ValueError("PUBCHEM_CID column is missing.")

# Add fields for reported results and provenance
new_columns = [
    "Solubility_Reported_Text",
    "Solubility_Source",
    "Solubility_Record_URL",
    "Solubility_Lookup_Status",
]

for column in new_columns:
    if column not in df.columns:
        df[column] = pd.NA


# --------------------------------------------------
# 2. EXTRACT TEXT FROM PUBCHEM ANNOTATIONS
# --------------------------------------------------

def value_to_text(value):
    """Convert a PubChem annotation value into readable text."""

    if isinstance(value, str):
        return value

    if isinstance(value, (int, float)):
        return str(value)

    if isinstance(value, list):
        return "; ".join(
            text for text in
            (value_to_text(item) for item in value)
            if text
        )

    if isinstance(value, dict):

        if "StringWithMarkup" in value:
            return "; ".join(
                item.get("String", "")
                for item in value["StringWithMarkup"]
                if isinstance(item, dict)
                and item.get("String")
            )

        pieces = []

        for key in [
            "String",
            "Number",
            "Unit",
            "Boolean",
            "Integer",
            "Description",
        ]:
            if key in value:
                text = value_to_text(value[key])
                if text:
                    pieces.append(f"{key}: {text}")

        if pieces:
            return "; ".join(pieces)

        return json.dumps(value, ensure_ascii=False)

    return ""


def find_annotation_values(data):
    """Find named values inside the returned JSON."""

    found = []

    def walk(obj):
        if isinstance(obj, dict):

            if "Value" in obj:
                label = obj.get("Name") or obj.get("TOCHeading")
                text = value_to_text(obj["Value"])

                if text:
                    if label:
                        found.append(f"{label}: {text}")
                    else:
                        found.append(text)

            for key, value in obj.items():
                if key != "Value":
                    walk(value)

        elif isinstance(obj, list):
            for item in obj:
                walk(item)

    walk(data)

    # Remove exact duplicates while keeping the original order
    return list(dict.fromkeys(found))


# --------------------------------------------------
# 3. LOOK UP SOLUBILITY ANNOTATIONS
# --------------------------------------------------

session = requests.Session()
session.headers.update({
    "User-Agent": "ERa-Agonist-Student-Research/1.0"
})

processed_this_run = 0
found_count = 0
no_result_count = 0
error_count = 0

print("\nStarting lookup...")
print("Total chemicals in file:", len(df))

for idx in df.index:

    if MAX_RECORDS is not None and processed_this_run >= MAX_RECORDS:
        break

    old_status = df.at[idx, "Solubility_Lookup_Status"]

    # Skip records already processed during an earlier run
    if pd.notna(old_status) and not str(old_status).startswith("Retry"):
        continue

    cid = df.at[idx, "PUBCHEM_CID"]

    if pd.isna(cid) or not str(cid).strip():
        df.at[idx, "Solubility_Lookup_Status"] = "Missing PubChem CID"
        processed_this_run += 1
        continue

    cid = str(cid).strip().split(".")[0]

    if not cid.isdigit():
        df.at[idx, "Solubility_Lookup_Status"] = "Invalid PubChem CID"
        processed_this_run += 1
        continue

    url = (
        "https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/"
        f"data/compound/{cid}/JSON?heading=Solubility"
    )

    df.at[idx, "Solubility_Record_URL"] = (
        f"https://pubchem.ncbi.nlm.nih.gov/compound/{cid}"
    )

    try:
        response = session.get(url, timeout=25)

        if response.status_code == 200:
            data = response.json()
            annotations = find_annotation_values(data)

            if annotations:
                df.at[idx, "Solubility_Reported_Text"] = (
                    " | ".join(annotations)
                )
                df.at[idx, "Solubility_Source"] = "PubChem PUG-View"
                df.at[idx, "Solubility_Lookup_Status"] = (
                    "Annotation found; review units and source"
                )
                found_count += 1
            else:
                df.at[idx, "Solubility_Lookup_Status"] = (
                    "No solubility values extracted"
                )
                no_result_count += 1

        elif response.status_code == 404:
            df.at[idx, "Solubility_Lookup_Status"] = (
                "No solubility annotation returned (404)"
            )
            no_result_count += 1

        elif response.status_code == 429 or response.status_code >= 500:
            df.at[idx, "Solubility_Lookup_Status"] = (
                f"Retry later (HTTP {response.status_code})"
            )
            error_count += 1

        else:
            df.at[idx, "Solubility_Lookup_Status"] = (
                f"HTTP {response.status_code}; review response"
            )
            error_count += 1

    except (requests.RequestException, ValueError) as error:
        df.at[idx, "Solubility_Lookup_Status"] = (
            f"Retry later: {str(error)[:120]}"
        )
        error_count += 1

    processed_this_run += 1

    if processed_this_run % 5 == 0:
        df.to_csv(progress_file, index=False)
        print(
            f"Processed {processed_this_run} this run; "
            f"found {found_count} records with annotations."
        )

    # Keep requests below PubChem's stated maximum rate.
    time.sleep(0.4)

# Save progress without overwriting your original template
df.to_csv(progress_file, index=False)
df.to_csv(final_file, index=False)

print("\nLOOKUP FINISHED")
print("Records processed this run:", processed_this_run)
print("Records with annotations:", found_count)
print("No results extracted:", no_result_count)
print("Errors to review or retry:", error_count)
print("\nProgress file:", progress_file)
print("Results file:", final_file)

print(
    "\nNOTE: Reported annotations can contain different units "
    "and source types. Review them before combining values."
)
