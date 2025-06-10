# DriverSub-SVM

**DriverSub-SVM** is a machine learning framework for cancer subtype classification.  
It integrates the heterogeneity of personalized driver genes with the commonalities of global driver genes, enabling the identification of subtype-specific biomarkers.  
This comprehensive approach enhances the accuracy and interpretability of cancer classification.

---

## 1. Environment Requirements

- Python 3.9
- Required libraries:
  - `numpy`
  - `pandas`
  - `scikit-learn`
  - `scipy`
  - `tqdm`
  - `optuna`

---

## 2. Input Data

We provide processed data for three cancer types: **BRCA**, **THCA**, and **STAD**.   

BRCA data clarification:
The PAM50 subtype dataset contains 1218 breast cancer samples, while the multi-omics data (mutation, CNV, expression) includes 1101 samples. 
After intersecting these two sources, 1089 samples with both subtype labels and complete multi-omics features were retained for model training and evaluation. 
This number is consistent with Table 1 in the manuscript, the specific dimensions of the other two cancer datasets are provided in the manuscript.

### Files:

- **`InfluenceGraph_Sig_Fls_inter_vstCNV`**  
  A protein functional interaction network constructed by integrating SignaLink3 and Reactome.  
  Gene pairs with concurrent interactions in both networks are retained.

- **`MutMatrix_Sig_Fls_inter_vstCNV`**  
  Combines somatic mutation and CNV data.  
  A gene is labeled as "aberrant" (value = 1) if it has a mutation or CNV value of 1 or -1 in a sample; otherwise, it is 0.

- **`OutMatrix_Sig_Fls_inter_vstCNV`**  
  VST-normalized transcriptomic expression values are converted into z-scores.  
  Genes with absolute z-score ≥ 2δ are marked as abnormally expressed (1), otherwise 0.

- **Subtype Labels:**

  | Cancer Type | Subtype Definition |
  |-------------|--------------------|
  | BRCA        | PAM50: LumA, LumB, Her2, Basal, Normal |
  | THCA        | Histological types: Classical, Follicular, Tall cell, Other |
  | STAD        | Expression-based: GS, CIN, EBV, MSI |

---

## 3. Source Code Description

| File Name               | Function |
|------------------------|----------|
| `Data.py`              | Loads input datasets. |
| `Random_walk.py`       | Implements a random walk to assess interactions between mutated and dysregulated genes. |
| `personalized.py`      | Applies the BPR algorithm to personalize gene ranking based on mutation profiles. |
| `global_driver.py`     | Uses the Condorcet method to identify global driver genes across samples. |
| `model_training_cross.py` | Trains an OAO-MSVM classifier for subtype prediction and performs evaluation. |

---

## 4. How to Run

Run the following command from the `Code` directory: python model_training_cross.py

This script will:

Perform k-fold cross-validation

BRCA: 10-fold (due to larger sample size)

THCA & STAD: 5-fold (due to smaller sample sizes)

Output average evaluation metrics (accuracy, precision, recall, F1-score)

Generate SHAP values and KEGG pathway input data for downstream interpretability analysis.

## 5.Notes
This project supports reproducibility. All datasets and preprocessing steps used for BRCA, THCA, and STAD are included or described in this repository.
See the /data/ and /Code/ directories for more details.

For questions or issues, please refer to the supplementary materials or contact the authors.

