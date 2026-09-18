# Cobber Learn Chemistry Projects

This repository contains small Python projects for chemistry, data analysis, machine learning, and model evaluation.

## Organized project layout

```text
projects/
├── chemistry/
│   ├── alkane_boiling_points.py
│   ├── molecule_explorer.py
│   └── pubchem_fetcher.py
├── data_analysis/
│   └── making_data_whole.py
├── model_evaluation/
│   └── error_metrics.py
└── hello_world/
    └── hello_world.py
```

## Projects

- **Chemistry**: alkane boiling-point visualization, RDKit molecular descriptors, and PubChem lookup.
- **Data analysis**: Titanic age prediction with a random-forest model.
- **Model evaluation**: MAE, MSE, R², and diagnostic plots comparing actual and predicted values.
- **Hello World**: introductory Python example.

## Running the scripts

Install the dependencies required by the project you want to run:

```bash
python -m pip install matplotlib numpy pandas scikit-learn seaborn pubchempy rdkit
```

Examples:

```bash
python projects/chemistry/alkane_boiling_points.py
python projects/chemistry/molecule_explorer.py
python projects/chemistry/pubchem_fetcher.py
python projects/data_analysis/making_data_whole.py
python projects/model_evaluation/error_metrics.py
python projects/hello_world/hello_world.py
```

The original files are still present in their legacy locations so that existing links and history are not broken. The `projects/` tree is now the canonical organized layout.
