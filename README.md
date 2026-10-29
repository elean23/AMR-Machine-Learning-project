# **Predicting Antimicrobial Resistance (AMR) Burden in Livestock**
## **Author:** Eleojo Nwokeoji
## **Programme:** CFG + Masters, AI/ML 

---

## Overview

Antimicrobial resistance in food-producing animals is a One Health priority. Laboratory antibiotic susceptibility testing (AST) is slow, labour-intensive and inconsistently reported across surveillance pipelines. This project asks:

> Can supervised machine learning models accurately predict the multidrug resistance (MDR) status and resistance class count of livestock bacterial isolates using epidemiological metadata and non-AMR genomic background features, across major pathogen species in a multi-year global surveillance dataset?

Two targets are predicted:

| Target | Type | Definition |
| --- | --- | --- |
| `n_drug_classes` | Regression | Number of distinct antibiotic drug classes (0 to 14) for which an isolate carries resistance determinants |
| `mdr_label` | Binary classification | 1 if the isolate carries determinants for 3 or more drug classes, otherwise 0 |

**Leakage control:** AMR genotype is the source of both targets, so it is excluded from the predictor matrix.

### Important caveat

Only about 1.74% of isolates (1,814 of 104,211) have usable AST phenotypes, and none of the 6,471 UK livestock isolates do. The targets are therefore **genotype-derived proxies**, not lab-confirmed phenotypes. On the 1,357 AST-labelled livestock isolates, the genotype-derived MDR label agreed with AST in 77.9% of cases (recall ~99%, precision ~59%), meaning it tends to over-call resistance. Model outputs should be read as **genotypic resistance risk** and used as a prioritisation or triage signal alongside AST, not as a replacement for it.

---

## Data

- **Source:** [NCBI Pathogen Detection, Isolates Browser](https://www.ncbi.nlm.nih.gov/pathogens/isolates)
- **Raw size:** 104,211 isolates (104,208 after removing duplicate BioSamples), 22 raw columns
- **Organisms:** *E. coli*/*Shigella*, *Salmonella enterica*, *Campylobacter jejuni*, *Staphylococcus aureus*
- **Coverage:** 14 countries
- **Modelling subset:** 88,837 isolates from four livestock hosts (cattle, poultry, pig, sheep/goat). Wild boar/ambiguous *Sus scrofa*, horse and rabbit records were excluded.
- **Access:** the notebook downloads the exported CSV from Google Drive via the `DATA_URL` variable in Section 2.1. A small number of rows (<0.1%) with malformed quoting are skipped on load.

---

## Repository contents

| File | Description |
| --- | --- |
| `Eleojo_A_N_Assignment_1_2_amrMLM_Livestock.ipynb` | Full pipeline: Assignment 1 (cleaning, EDA, encoding) and Assignment 2 (modelling, evaluation, graph analysis) |
| `Eleojo_A_N_Combined_Assignment_1_2_AMR_Report_Final.docx` | Combined written report |


### Files generated when the notebook runs

**Assignment 1 outputs**
- `amr_livestock_features_clean.csv`: human-readable modelling table
- `amr_livestock_features_encoded.csv`: one-hot encoded full table
- `X_train.csv`, `X_test.csv`, `y_train.csv`, `y_test.csv`: scaled 80/20 split

**Assignment 2 outputs**
- `phase2_classification_baselines.csv`
- `phase2_classification_validation_comparison.csv`
- `phase2_classification_test_results.csv`
- `phase2_regression_validation_results.csv`
- `phase2_regression_test_results.csv`
- `phase2_graph_feature_validation_comparison.csv`
- `phase2_gnn_comparison.csv` (only if PyTorch Geometric is available)

---

## Pipeline

### Assignment 1: data preparation and EDA

1. **Load and inspect** the NCBI export; strip stray `#` from column headers; parse dates to year.
2. **Handle missing data:** drop AST phenotypes (98.3% missing), Serovar (71.3% missing) and uninformative metadata; recode missing isolation type/source to `Unknown` and missing SNP cluster to `Unclustered`; treat missing gene lists as zero genes.
3. **Remove duplicates** on BioSample.
4. **Engineer targets:** map AMR gene names to drug classes via a regex rule set based on AMRFinderPlus/CARD naming (e.g. `bla` to beta-lactams, `tet` to tetracyclines, `aac`/`aph` to aminoglycosides), then count classes per isolate.
5. **Engineer features:** host category, isolation type, sample source detail, country, organism group, year, `n_virulence_genes`, `n_stress_genes`.
6. **EDA:** target distribution, MDR rate by host/organism/sampling context, correlation analysis, outlier checks.
7. **Encode:** one-hot encode low-cardinality categoricals; keep the top 50 SNP clusters plus `Minor/Other`; add presence/absence columns for the top 100 virulence and top 50 stress genes (236 encoded feature columns in total).
8. **Split and scale:** 80/20 stratified on `organism_group` + `mdr_label`; scaler fitted on training data only.

### Assignment 2: modelling and evaluation

1. **Target validation** against the AST-labelled subset.
2. **70/15/15 train/validation/test split**, stratified on `organism_group` + `mdr_label`. The test set stays locked until final evaluation.
3. **Classification baselines:** Logistic Regression, Random Forest, HistGradientBoosting, MLP, XGBoost, CatBoost.
4. **Hyperparameter tuning:** `RandomizedSearchCV` on Random Forest and HistGradientBoosting (cross-validation inside the training set only).
5. **Model selection** by validation F1 before the test set is inspected.
6. **Interpretability:** native feature importance or permutation importance on the validation set.
7. **Regression** of `n_drug_classes` (selection by validation MAE).
8. **Gene co-occurrence network:** descriptive full-population graph, plus a training-only graph-centrality feature.
9. **Graph neural network (experimental):** per-isolate gene graphs with training-only edges, compared against a gene-set ablation (no message passing) and the Random Forest on identical test isolates.

---

## Key results

Figures below are from the previously executed run and are reproduced from the report. Re-running the notebook may shift them slightly.

### Dataset findings

| Host | N | MDR prevalence | Mean drug classes |
| --- | --- | --- | --- |
| Pig | 17,315 | 79.8% | 5.44 |
| Poultry | 25,425 | 52.6% | 3.56 |
| Cattle | 43,711 | 36.3% | 2.67 |
| Sheep/Goat | 2,386 | 33.9% | 2.45 |

Clinical isolates and isolates with unknown isolation type show much higher MDR rates (about 71%) than environmental/other isolates (36.9%), which reflects sampling context as well as biology. `n_stress_genes` correlates moderately with `n_drug_classes` (r = 0.40); `n_virulence_genes` shows almost no linear relationship (r = -0.07).

### Classification (MDR vs non-MDR), test set

| Model | Accuracy | F1 | ROC-AUC | PR-AUC |
| --- | --- | --- | --- | --- |
| **Random Forest (untuned, selected)** | 0.887 | 0.883 | 0.957 | 0.961 |
| HistGradientBoosting (tuned) | 0.884 | ~0.88 | 0.957 | 0.962 |
| Random Forest (tuned) | 0.876 | ~0.87 | 0.953 | 0.957 |
| MLP | 0.876 | ~0.87 | 0.951 | 0.956 |
| HistGradientBoosting (untuned) | 0.874 | ~0.87 | 0.950 | 0.955 |
| Logistic Regression | 0.851 | ~0.84 | 0.930 | 0.938 |

### Regression (`n_drug_classes`), test set

| Model | MAE | RMSE | R² |
| --- | --- | --- | --- |
| **Random Forest Regressor** | 0.620 | 1.012 | 0.876 |
| MLP Regressor | 0.754 | 1.098 | 0.855 |
| HistGradientBoosting Regressor | 0.810 | 1.142 | 0.843 |
| Linear Regression | 1.010 | 1.377 | 0.771 |

### Other findings

- Specific stress and virulence gene identities carry more predictive signal than simple gene counts.
- Hub genes in the descriptive network (e.g. `invA`, `avrA`, `iroB`, `iroC`) largely reflect *Salmonella* lineage structure rather than AMR co-selection.
- A single graph-centrality feature did not improve Random Forest performance.
- GNN vs gene-set ablation vs Random Forest: **results to be added from the fixed-split notebook run** (see `phase2_gnn_comparison.csv`).

---

## Getting started

### Option 1: Google Colab (recommended)

1. Open the notebook in Colab.
2. Run the cells top to bottom. The raw data downloads automatically.
3. Uncomment the install cells when prompted:
   - `!pip -q install xgboost catboost`
   - `!pip -q install torch-geometric` (needed for the GNN section only)

### Option 2: Local

```bash
git clone <your-repo-url>
cd <your-repo-name>

python -m venv .venv
source .venv/bin/activate
# For Windows: .venv\Scripts\activate

pip install pandas numpy matplotlib seaborn plotly scikit-learn \
            xgboost catboost networkx jupyter

# For the GNN section:
pip install torch torch-geometric

```

### Run order

The notebook must be run in order. Assignment 2 reads the CSVs written at the end of Assignment 1 and raises a `FileNotFoundError` if they are missing. The GNN section is skipped gracefully if PyTorch Geometric is unavailable.

All random seeds are fixed (`random_state=42`).

---

## Limitations

- Targets are genotype-derived proxies; AST coverage is sparse, especially for UK livestock.
- The gene-to-drug-class mapping is a curated rule set, not the official AMRFinderPlus catalogue.
- `Year` reflects NCBI submission date, not necessarily collection date.
- Sampling is uneven across hosts, organisms, countries and isolation contexts. Small subgroups (*S. aureus*, sheep/goat) give less stable estimates.
- The top-gene vocabulary was chosen on the full modelling population in Assignment 1; strict leakage isolation would select it within the training partition.
- GNN conclusions depend on graph construction, edge weighting and optimisation settings.

## Future work

- Validate the target on a larger, geographically independent AST dataset.
- Select the top-gene vocabulary strictly within the training partition.
- Build organism-specific and accessory-gene-focused networks.
- Add richer graph structure (plasmids, mobile genetic elements, functional annotations).
- Assess probability calibration and decision thresholds for surveillance screening.
- Evaluate on an independent external cohort.

---

## References

- Feldgarden, M. et al. (2021) AMRFinderPlus and the Reference Gene Catalog facilitate examination of the genomic links among antimicrobial resistance, stress response, and virulence. *Scientific Reports*, 11, 12728.
- Grinsztajn, L., Oyallon, E. and Varoquaux, G. (2022) Why do tree-based models still outperform deep learning on typical tabular data? *NeurIPS*, 35, 507-520.
- NCBI Pathogen Detection. National Library of Medicine. https://www.ncbi.nlm.nih.gov/pathogens/
- Nguyen, H. A. et al. (2026) AMR-GNN: a multi-representation graph neural network framework to enable genomic antimicrobial resistance prediction. *Nature Communications*, 17(1), 3555.
- UK-VARSS (2024) Veterinary Antibiotic Resistance and Sales Surveillance Report (UK-VARSS 2023). Veterinary Medicines Directorate.
- World Health Organization (2015) Global Action Plan on Antimicrobial Resistance.

## License

NCBI Pathogen Detection data are public; check NCBI's data-use policy before redistributing derived files.


