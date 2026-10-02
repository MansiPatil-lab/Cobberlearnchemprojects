import pandas as pd

annotations = pd.read_excel(
    "../data/assay_annotations_invitrodb_v4_3_AUG2025.xlsx"
)

# Show every endpoint belonging to the ERa BLA Agonist assay
result = annotations[
    annotations["assay_name"]
    .astype(str)
    .str.contains("TOX21_ERa_BLA_Agonist", case=False, na=False)
]

print(
    result[
        ["aid", "asid", "assay_name", "assay_component_endpoint_name"]
    ].to_string(index=False)
)