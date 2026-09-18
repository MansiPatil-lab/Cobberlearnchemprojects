# Project guide

The repository was reviewed and a beginner-friendly canonical layout was added without deleting or overwriting any existing file.

```text
src/
├── chemistry/
│   ├── alkane_boiling_points.py
│   ├── molecule_explorer.py
│   └── pubchem_fetcher.py
├── data_analysis/
│   ├── making_data_whole.py
│   └── molecule_solubility_dataframe.py
├── hello_world/
│   └── hello_world.py
├── modeling/
│   └── decision_tree_classifier.py
└── model_evaluation/
    └── error_metrics.py

data/       Dataset notes and a place for future input datasets.
plots/       Plot organization notes; existing figures remain in their assignments.
results/     Reserved for analysis outputs.
docs/        Repository documentation.
```

## Existing files

The repository already contained an organized `projects/` tree. It remains unchanged and continues to be usable. The new `src/` tree contains preserved copies of the maintained scripts, grouped by purpose. Older folders, standalone scripts, generated images, SVGs, and IDE settings were intentionally left untouched.

## Running examples

From the repository root:

```bash
python src/hello_world/hello_world.py
python src/chemistry/alkane_boiling_points.py
python src/data_analysis/molecule_solubility_dataframe.py
python src/modeling/decision_tree_classifier.py
python src/model_evaluation/error_metrics.py
```

Install only the dependencies needed for the example you want to run. The existing root `README.md` contains the original dependency list and project commands.
