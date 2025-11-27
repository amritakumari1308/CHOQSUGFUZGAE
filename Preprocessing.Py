## -*- coding: -*-
## FuMOGAE , GAE is applied to six omic modalities (CNA, GSE, DNA-Methylation, miRNA, Mutation, and
## Co-expression) after hybrid Boruta + mRMR feature selection. Latent embeddings extracted from these
## six GAEs are concatenated with the Clinical (CLN) features to form a unified representation.
## The final breast cancer subtype classification is performed using a hybrid Choquet–Sugeno fuzzy integral.
#
#
!pip install category_encoders
!pip install torch_geometric
!pip install pymrmr
# Install required dependencies
!pip install numpy scipy scikit-learn matplotlib h5py torch torchvision torchaudio
!pip install imbalanced-learn
!pip install boruta
pip install ranger-adabelief
!pip install mrmr_selection
pip install imbalanced-learn optuna torch xgboost scikit-learn pandas joblib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.nn import GCNConv, GAE
from torch_geometric.data import Data
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from boruta import BorutaPy
from sklearn.neighbors import kneighbors_graph
from torch_geometric.utils import from_scipy_sparse_matrix
import torch.nn as nn
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import re

# Load the datasets
cna_data = pd.read_csv("/content/drive/MyDrive/Colab_Notebooks/breast_cancer/new_try/data/cna_1253.csv",header=None).dropna(how="all")
clinical_data = pd.read_csv("/content/drive/MyDrive/Colab_Notebooks/breast_cancer/new_try/data/clinical_1253.csv", header=None)
gse_data = pd.read_csv("/content/drive/MyDrive/Colab_Notebooks/breast_cancer/new_try/data/gse_1253.csv", header=None)
Dna_data = pd.read_csv("/content/drive/MyDrive/Colab_Notebooks/breast_cancer/new_try/data/dna_1253.csv", header=None)
mi_rna_data=pd.read_csv("/content/drive/MyDrive/Colab_Notebooks/breast_cancer/new_try/data/mi_rna_1253.csv", header=None)
mut_data=pd.read_csv("/content/drive/MyDrive/Colab_Notebooks/breast_cancer/new_try/data/mutation_1253.csv", header=None)
coexp_data=pd.read_csv("/content/drive/MyDrive/Colab_Notebooks/breast_cancer/new_try/data/module_eigengenes_per_sample (1).csv",header=None)
cna_data.head()
# Converted Label and Patient ID to integer value

import pandas as pd


# Define the mapping of labels to numbers
label_mapping = {
    'LumB': 0,
    'Normal': 1,
    'Her2': 2,
    'LumA': 3,
    'Basal': 4

}

clinical_data.iloc[:, -1] = clinical_data.iloc[:, -1].map(label_mapping)
cna_data.iloc[:, -1] = cna_data.iloc[:, -1].map(label_mapping)
gse_data.iloc[:, -1] = gse_data.iloc[:, -1].map(label_mapping)
Dna_data.iloc[:, -1] = Dna_data.iloc[:, -1].map(label_mapping)
mi_rna_data.iloc[:, -1] = mi_rna_data.iloc[:, -1].map(label_mapping)
mut_data.iloc[:, -1] = mut_data.iloc[:, -1].map(label_mapping)
coexp_data.iloc[:,-1]=coexp_data.iloc[:,-1].map(label_mapping)


# Save the updated DataFrame to a new CSV file
clinical_data.to_csv('clinical_data_with_labels_as_numbers.csv', index=False)
cna_data.to_csv('cna_data_with_labels_as_numbers.csv', index=False)
gse_data.to_csv('gse_data_with_labels_as_numbers.csv', index=False)
Dna_data.to_csv('Dna_data_with_labels_as_numbers.csv', index=False)
mi_rna_data.to_csv('mi_rna_data_with_labels_as_numbers.csv', index=False)
mut_data.to_csv('mut_data_with_labels_as_numbers.csv', index=False)
coexp_data.to_csv('coexp_data_with_labels_as_numbers.csv',index=False)
coexp_data.head()
# Converted Label and Patient ID to integer value

import pandas as pd

clinical_data.iloc[:, 0] = clinical_data.iloc[:, 0].astype(str).str.replace("MB-", "", regex=False).astype(int)
cna_data.iloc[:, 0] = cna_data.iloc[:, 0].astype(str).str.replace("MB-", "", regex=False).astype(int)
gse_data.iloc[:, 0] = gse_data.iloc[:, 0].astype(str).str.replace("MB-", "", regex=False).astype(int)
Dna_data.iloc[:, 0] = Dna_data.iloc[:, 0].astype(str).str.replace("MB-", "", regex=False).astype(int)
mi_rna_data.iloc[:, 0] = mi_rna_data.iloc[:, 0].astype(str).str.replace("MB-", "", regex=False).astype(int)
mut_data.iloc[:, 0] = mut_data.iloc[:, 0].astype(str).str.replace("MB-", "", regex=False).astype(int)
coexp_data.iloc[:,0]=coexp_data.iloc[:, 0].astype(str).str.replace("MB-", "", regex=False).astype(int)


# Save the updated DataFrame to a new CSV file
clinical_data.to_csv('clinical_data_all_int.csv', index=False)
cna_data.to_csv('cna_data_all_int.csv', index=False)
gse_data.to_csv('gse_data_all_int.csv', index=False)
Dna_data.to_csv('Dna_data_all_int.csv', index=False)
mi_rna_data.to_csv('mi_rna_data_all_int.csv', index=False)
mut_data.to_csv('mut_data_all_int.csv', index=False)
coexp_data.to_csv('coexp_data_all_int.csv',index=False)
cna_data.head()

import pandas as pd


# Count the number of samples for each label
label_counts = mut_data.iloc[:, -1].value_counts()

print(label_counts)
mi_rna_data=pd.read_csv("mi_rna_data_all_int.csv")
mi_rna_data.head()
--------miRNA Preprocessing ------
import pandas as pd
import numpy as np
from pathlib import Path
from statsmodels.stats.multitest import multipletests
from scipy.stats import kruskal

# -------- Paths --------
in_path = Path("/content/mi_rna_data_all_int.csv")
out_matrix = Path("miRNA_selected_features.csv")
out_names  = Path("miRNA_selected_feature_names.txt")
out_log    = Path("miRNA_preproc_log.txt")

# -------- Robust CSV loader (handles missing/wrong header) --------
def load_mirna_csv(path: Path) -> pd.DataFrame:
    df_try = pd.read_csv(path)
    # Heuristic: if last column header is a subtype string or first col header looks like a patient ID,
    # it's likely the file's first row was treated as header.
    suspicious_headers = {"LumA", "LumB", "Basal", "Her2", "Normal"}
    bad_header = (df_try.columns[-1] in suspicious_headers) or df_try.columns[0].startswith("MB-")
    if bad_header:
        df = pd.read_csv(path, header=None)
        # Name columns: first=PatientID, last=Subtype, middle = miR_0..
        n_cols = df.shape[1]
        cols = (["PatientID"] + [f"miR_{i}" for i in range(1, n_cols-1)] + ["Subtype"])
        df.columns = cols
    else:
        df = df_try
        # Standardize expected names if present
        if df.columns[0].lower() not in ["patientid", "patient_id", "id"]:
            df = df.rename(columns={df.columns[0]: "PatientID"})
        if df.columns[-1].lower() not in ["subtype", "label", "pam50"]:
            df = df.rename(columns={df.columns[-1]: "Subtype"})
    return df

mi_rna_data = load_mirna_csv(in_path)

# Coerce expression columns to numeric (keep PatientID/Subtype as is)
expr_cols = mi_rna_data.columns[1:-1]
for c in expr_cols:
    mi_rna_data[c] = pd.to_numeric(mi_rna_data[c], errors="coerce")

# Drop rows with missing subtype
mi_rna_data = mi_rna_data[~mi_rna_data["Subtype"].isna()].copy()

# -------- Step 1: RPM presence filter (paper: >= 0.01 RPM in >= 30% samples) --------
# Since values appear already normalized / z-scored, we approximate using the given threshold directly.
presence = (mi_rna_data[expr_cols] >= 0.01).sum(axis=0) / mi_rna_data.shape[0]
keep_presence = presence.index[presence >= 0.30].tolist()

# If this is too strict (e.g., keeps very few), relax slightly but log it.
relaxed = False
if len(keep_presence) < 50:
    presence = (mi_rna_data[expr_cols] > 0).sum(axis=0) / mi_rna_data.shape[0]
    keep_presence = presence.index[presence >= 0.30].tolist()
    relaxed = True

mi_rna_data_filt = mi_rna_data[["PatientID"] + keep_presence + ["Subtype"]].copy()

# -------- Step 2: Differential expression proxy (paper uses DESeq2 tumor vs normal tissue)
# We approximate using Kruskal-Wallis across PAM50 subtypes and apply FDR<=0.01
labels = mi_rna_data_filt["Subtype"].astype(str)
groups = labels.unique().tolist()

pvals = []
genes = keep_presence
for g in genes:
    # Build groups for Kruskal (require at least 2 non-empty groups)
    samples = [mi_rna_data_filt.loc[labels==lab, g].dropna().values for lab in groups]
    # Only test if we have at least 2 groups with >1 sample
    valid = [s for s in samples if s.size > 1]
    if len(valid) >= 2:
        try:
            stat, p = kruskal(*valid)
        except Exception:
            p = 1.0
    else:
        p = 1.0
    pvals.append(p)

pvals = np.array(pvals, dtype=float)
_, qvals, _, _ = multipletests(pvals, alpha=0.01, method="fdr_bh")

sel_mask = qvals <= 0.01
selected_genes = list(np.array(genes)[sel_mask])

# If none pass FDR, fallback to top by variance after presence filter
if len(selected_genes) == 0:
    variances = mi_rna_data_filt[genes].var(axis=0).sort_values(ascending=False)
    selected_genes = variances.index.tolist()

# Paper retained 343 miRNAs; if we have more, take top 343 by Kruskal H-stat (proxy: -log10 p)
if len(selected_genes) > 343:
    # Order by p-value ascending
    order = np.argsort(pvals[sel_mask])
    top_idx = order[:343]
    selected_genes = list(np.array(genes)[sel_mask][top_idx])
elif len(selected_genes) < 343:
    # If fewer than 343, pad with next best p-values until 343 or run out
    remaining = list(np.array(genes)[~sel_mask])
    order = np.argsort(pvals[~sel_mask])
    pad = [remaining[i] for i in order[:max(0, 343-len(selected_genes))]]
    selected_genes = selected_genes + pad

# Deduplicate just in case
selected_genes = list(dict.fromkeys(selected_genes))

# -------- Save outputs --------
mi_rna_data_out = mi_rna_data_filt[["PatientID"] + selected_genes + ["Subtype"]].copy()
mi_rna_data_out.to_csv(out_matrix, index=False)

with open(out_names, "w") as f:
    for g in selected_genes:
        f.write(f"{g}\n")

with open(out_log, "w") as f:
    f.write("miRNA preprocessing  log\n")
    f.write(f"Input samples: {mi_rna_data.shape[0]}\n")
    f.write(f"Input miRNA features: {len(expr_cols)}\n")
    f.write(f"Presence filter cutoff: >=0.01 in >=30% (relaxed to >0 if needed={relaxed})\n")
    f.write(f"Kept after presence filter: {len(keep_presence)}\n")
    f.write(f"Selected after FDR<=0.01 + padding: {len(selected_genes)} (target=343)\n")
    f.write(f"Groups: {groups}\n")

(mi_rna_data.shape, len(expr_cols), len(keep_presence), len(selected_genes), out_matrix.as_posix(), out_names.as_posix(), out_log.as_posix())
mi_rna_data=pd.read_csv("/content/miRNA_selected_features.csv")
mi_rna_data.head()
---------MUTATION Preprocesing------------
print(mut_data.head())
# convert mutation to integer data

# Convert a single mutation cell into 0/1
ZERO_TOKENS = {"0", "", "na", "nan", "none", "-", "."}

def parse_mut_cell(cell) -> int:
    if pd.isna(cell):
        return 0
    s = str(cell).strip()
    if s.lower() in ZERO_TOKENS:
        return 0
    # If it looks like a pure number, keep it as int
    if re.fullmatch(r'[+-]?\d+(\.\d+)?', s):
        return int(float(s))
    # Otherwise treat any string mutation as "present"
    return 1

# Apply to all feature columns (skip first=PatientID and last=label)
def preprocess_mutation_features(mut_df: pd.DataFrame) -> pd.DataFrame:
    df = mut_df.copy()
    feat_cols = df.columns[1:-1]   # assuming [ID | features... | label]
    df[feat_cols] = df[feat_cols].applymap(parse_mut_cell).fillna(0).astype(int)
    return df

# ---- Usage ----
# mut_data = pd.read_csv("mutation.csv")

mut_data = preprocess_mutation_features(mut_data)

# Save
mut_data.to_csv("mutation_features_preprocess.csv", index=False)
mut_data.head()
Dna_data.head(5)
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# ----------------------------
# 0) Put your 7 DataFrames here
# ----------------------------
# clinical_data, gse_data, cna_data, Dna_data, mi_rna_data, mut_data, coexp_data
datasets = {
    "clinical": clinical_data,
    "gse": gse_data,
    "cna": cna_data,
    "dna": Dna_data,
    "mirna": mi_rna_data,
    "mutation": mut_data,
    "COEXP": coexp_data,  # now included
}

# ---------------------------------------
# 1) Extract labels & check consistency
# ---------------------------------------
def get_labels(df: pd.DataFrame) -> np.ndarray:
    return df.iloc[:, -1].to_numpy()

labels = {name: get_labels(df) for name, df in datasets.items()}

# Choose clinical as reference (replace if you prefer another)
ref_name = "clinical"
y_ref = labels[ref_name]

def report_label_inconsistency(labels_dict, reference, reference_name):
    ok = True
    for name, y in labels_dict.items():
        if name == reference_name:
            continue
        if len(y) != len(reference):
            ok = False
            print(f"[WARN] Length mismatch: {name}({len(y)}) vs {reference_name}({len(reference)})")
            continue
        neq = np.where(y != reference)[0]
        if len(neq) > 0:
            ok = False
            print(f"[WARN] Label mismatch against {reference_name} in: {name}")
            print(f"       #mismatched rows: {len(neq)} (showing first 10 indices): {neq[:10].tolist()}")
    return ok

is_consistent = report_label_inconsistency(labels, y_ref, ref_name)
if not is_consistent:
    pass
else:
    print("[OK] Labels are consistent across listed datasets.")

# ---------------------------------------------------------
# 2) Preprocess features: NaN handling + numeric encoding
# ---------------------------------------------------------
def preprocess_features(df: pd.DataFrame,
                        scaler="standard",
                        id_is_first_col=True,
                        label_is_last_col=True) -> np.ndarray:
    """
    - Keeps PatientID/Label columns untouched (excluded from X).
    - For non-numeric columns: factorize to integers (not dropped).
    - For numeric columns: fill NaN with column median.
    - For factorized columns: fill NaN with mode (0 after factorize).
    - Scale with StandardScaler (default) or MinMaxScaler.
    """
    X_df = df.copy()

    # Exclude ID + Label
    if id_is_first_col:
        X_df = X_df.iloc[:, 1:]
    if label_is_last_col:
        X_df = X_df.iloc[:, :-1]

    # Factorize non-numeric columns
    for col in X_df.columns:
        if not pd.api.types.is_numeric_dtype(X_df[col]):
            codes, uniques = pd.factorize(X_df[col].astype(str), sort=True)
            X_df[col] = codes.astype(float)

    # Fill NaNs: numeric median
    for col in X_df.columns:
        col_vals = pd.to_numeric(X_df[col], errors="coerce")
        med = np.nanmedian(col_vals.astype(float))
        if np.isnan(med):
            med = 0.0
        X_df[col] = np.where(np.isnan(col_vals), med, col_vals).astype(float)

    # Scale
    if scaler == "standard":
        scaler_obj = StandardScaler()
    elif scaler == "minmax":
        scaler_obj = MinMaxScaler()
    else:
        raise ValueError("scaler must be 'standard' or 'minmax'")
    X = scaler_obj.fit_transform(X_df.to_numpy().astype(np.float32)).astype(np.float32)
    return X

# You can choose per-modality scaler
scaler_choice = {
    "clinical": "minmax",
    "gse":      "standard",
    "cna":      "standard",
    "dna":      "standard",
    "mirna":    "standard",
    "mutation": "standard",
    "COEXP":    "standard",   # coexp scaling
}

# ---------------------------------------------------------
# 3) Generate feature matrices for all modalities
# ---------------------------------------------------------
X_mats = {}
for name, df in datasets.items():
    sc = scaler_choice.get(name, "standard")
    X_mats[name] = preprocess_features(df, scaler=sc)
    print(f"[{name}] X shape: {X_mats[name].shape}")

# If you also want the label vector to use downstream:
y = y_ref.astype(np.int64)
print("Label vector shape:", y.shape)
# --------------------------
# 3) Save final preprocessed CSVs
# --------------------------

# Rebuild each DataFrame with: PatientID | features... | Label
for name, df in datasets.items():
    # PatientID is first column, Label is last
    pid = df.iloc[:, 0].reset_index(drop=True)
    label = df.iloc[:, -1].reset_index(drop=True)

    # Features after preprocessing
    X = X_mats[name]
    feat_cols = [f"F{i}" for i in range(X.shape[1])]
    X_df = pd.DataFrame(X, columns=feat_cols)

    # Combine back
    final_df = pd.concat([pid, X_df, label], axis=1)
    final_df.columns = ["PatientID"] + feat_cols + ["Label"]

    # Save
    out_path = f"{name}_final_preprocessed.csv"
    final_df.to_csv(out_path, index=False)
    print(f"[SAVED] {name}: {out_path} | shape={final_df.shape}")
  clinical_data=pd.read_csv("clinical_final_preprocessed.csv")
cna_data=pd.read_csv("cna_final_preprocessed.csv")
gse_data=pd.read_csv("gse_final_preprocessed.csv")
Dna_data=pd.read_csv("dna_final_preprocessed.csv")
mi_rna_data=pd.read_csv("mirna_final_preprocessed.csv")
mut_data=pd.read_csv("mutation_final_preprocessed.csv")
coexp_data=pd.read_csv("COEXP_final_preprocessed.csv")

print(clinical_data.shape)
print(cna_data.shape)
print(gse_data.shape)
print(Dna_data.shape)
print(mi_rna_data.shape)
print(mut_data.shape)
print(coexp_data.shape)
clinical_data.head()
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# --------- 0) Prepare X (features) and y (labels) ----------
# Assumes: gse_data has PatientID in col 0 and Label in last col
X_gse_raw = gse_data.iloc[:, 1:-1].copy()
y_gse = gse_data.iloc[:, -1].to_numpy().astype(int)

# Ensure numeric + fill NaNs (median)
X_gse_raw = X_gse_raw.apply(pd.to_numeric, errors="coerce")
X_gse_raw = X_gse_raw.fillna(X_gse_raw.median(numeric_only=True))

# --------- 1) Scale ----------
scaler_gse = StandardScaler()
X_gse_scaled = scaler_gse.fit_transform(X_gse_raw.values.astype(np.float32))

# --------- 2) PCA (dimensionality reduction) ----------
n_features = X_gse_scaled.shape[1]
pca_k = min(400, n_features)  # cap at 400 or available features
pca = PCA(n_components=pca_k, random_state=42)
X_gse_reduced = pca.fit_transform(X_gse_scaled)
print("GSE Data after PCA Shape:", X_gse_reduced.shape)

# --------- 3) t-SNE (2D) ----------
# Choose a safe perplexity: must be < (n_samples-1)/3
n_samples = X_gse_reduced.shape[0]
max_perp = max(5, int((n_samples - 1) / 3) - 1)
perplexity = min(30, max_perp)  # typical default 30 but keep it valid
if perplexity < 5:
    perplexity = max(2, max_perp)  # fallback for very small datasets

tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, init="random", learning_rate="auto")
X_gse_tsne = tsne.fit_transform(X_gse_reduced)
print("GSE Data after t-SNE Shape:", X_gse_tsne.shape, "| perplexity:", perplexity)

# --------- 4) Plot ----------
plt.figure(figsize=(10, 8))
sns.scatterplot(
    x=X_gse_tsne[:, 0], y=X_gse_tsne[:, 1],
    hue=y_gse, palette='viridis', s=60, edgecolor=None, legend='full'
)
plt.title("t-SNE of GSE (after PCA)")
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.legend(title="Label", loc='best')
plt.tight_layout()
plt.show()
