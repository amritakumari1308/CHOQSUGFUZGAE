## -*- coding: -*-
## ChoqSugFuzGAE , GAE is applied to six omic modalities (CNA, GSE, DNA-Methylation, miRNA, Mutation, and
## Co-expression) after hybrid Boruta + mRMR feature selection. Latent embeddings extracted from these
## six GAEs are concatenated with the Clinical (CLN) features to form a unified representation.
## The final breast cancer subtype classification is performed using a hybrid Choquet–Sugeno fuzzy integral.


---------Graph Construction---------
import numpy as np
import torch
import matplotlib.pyplot as plt
import networkx as nx
from scipy.sparse import csr_matrix
from torch_geometric.data import Data
from torch_geometric.utils import from_scipy_sparse_matrix, to_undirected

def construct_knn_graph_pearson(
    data: np.ndarray,
    dataset_name: str,
    n_neighbors: int = 10,
    threshold: float | None = None,
    use_abs_for_knn: bool = True,   # choose neighbors by |r| but keep signed weights
    plot: bool = True,              # << turn on NetworkX visualization
    max_nodes_for_plot: int = 600,  # subsample for clearer plots
    layout_k: float | None = None,  # spring_layout k param (None lets networkx choose)
    edge_color_mode: str = "signed" # "signed" (coolwarm) or "abs" (Blues)
) -> Data:
    """
    Build a weighted, symmetric kNN patient–patient graph using Pearson correlation
    and (optionally) visualize it with NetworkX, coloring edges by weight.

    Returns
    -------
    torch_geometric.data.Data with x, edge_index, edge_weight
    """
    X = np.asarray(data, dtype=np.float32)
    N = X.shape[0]
    if N < 2:
        raise ValueError("Need at least 2 samples to build a graph.")
    k = int(min(max(1, n_neighbors), max(1, N - 1)))

    # --- Pearson sample–sample correlation (rows are samples) ---
    S = np.corrcoef(X)             # (N, N)
    np.fill_diagonal(S, 0.0)       # no self-edges
    S = np.nan_to_num(S, nan=0.0, posinf=0.0, neginf=0.0)
    S = np.clip(S, -1.0, 1.0)

    # --- kNN selection per node ---
    selector = np.abs(S) if use_abs_for_knn else S.copy()
    rows, cols, vals = [], [], []
    for i in range(N):
        idx = np.argpartition(selector[i], -k)[-k:]     # fast top-k (unordered)
        idx = idx[np.argsort(selector[i, idx])[::-1]]   # high -> low
        for j in idx:
            w = S[i, j]  # keep signed weight
            if threshold is not None:
                keep_score = abs(w) if use_abs_for_knn else w
                if keep_score < threshold:
                    continue
            rows.append(i); cols.append(j); vals.append(float(w))

    # --- Symmetrize: A = max(A, A^T) ---
    A = csr_matrix((vals, (rows, cols)), shape=(N, N))
    A = A.maximum(A.transpose())
    A.eliminate_zeros()

    # --- Convert to PyG ---
    edge_index, edge_weight = from_scipy_sparse_matrix(A)
    data_pyg = Data(
        x=torch.from_numpy(X),
        edge_index=edge_index,
        edge_weight=edge_weight
    )
    print(f"🔹 {dataset_name}: nodes={N}, undirected edges={edge_index.shape[1]//2} (stored directed={edge_index.shape[1]})")

    # --- Visualization (NetworkX, concept from your snippet) ---
    if plot:
        # Make sure we draw an undirected simple graph
        ei = to_undirected(edge_index).cpu().numpy()
        ew = edge_weight.detach().cpu().numpy().astype(float)

        # Build NX graph with a single undirected edge per pair; keep max |w|
        G = nx.Graph()
        G.add_nodes_from(range(N))
        weight_map = {}
        for (u, v), w in zip(ei.T, ew):
            a, b = (int(u), int(v)) if u <= v else (int(v), int(u))
            if a == b:
                continue
            if (a, b) in weight_map:
                if abs(w) > abs(weight_map[(a, b)]):
                    weight_map[(a, b)] = float(w)
            else:
                weight_map[(a, b)] = float(w)
        for (u, v), w in weight_map.items():
            G.add_edge(u, v, weight=w)

        # Subsample for readability if very large
        if G.number_of_nodes() > max_nodes_for_plot:
            degs = dict(G.degree())
            # keep 70% high-degree, 30% random among the rest
            keep_high = sorted(degs, key=degs.get, reverse=True)[: int(0.7*max_nodes_for_plot)]
            rest = [n for n in G if n not in keep_high]
            rng = np.random.default_rng(42)
            fill = []
            if len(keep_high) < max_nodes_for_plot and len(rest) > 0:
                fill = rng.choice(rest, size=max_nodes_for_plot-len(keep_high), replace=False).tolist()
            keep = set(keep_high) | set(fill)
            G = G.subgraph(keep).copy()

        # Layout
        pos = nx.spring_layout(G, seed=42, k=layout_k)

        # Edge colors
        edges = list(G.edges())
        if len(edges) == 0:
            print(f"[{dataset_name}] No edges to visualize after thresholding.")
        else:
            wts = np.array([G[u][v]["weight"] for u, v in edges], dtype=float)

            if edge_color_mode.lower() == "abs":
                colors = np.abs(wts)
                cmap = plt.cm.Blues
                vmin, vmax = 0.0, 1.0
            else:  # "signed"
                colors = wts
                cmap = plt.cm.coolwarm
                vmin, vmax = -1.0, 1.0

            plt.figure(figsize=(8, 8), dpi=160)
            nx.draw_networkx_nodes(G, pos, node_size=30, node_color="#444444", linewidths=0)
            ec = nx.draw_networkx_edges(
                G, pos,
                edge_color=colors,
                edge_cmap=cmap,
                edge_vmin=vmin,
                edge_vmax=vmax,
                width=1.0,
                alpha=0.6
            )
            plt.colorbar(ec, shrink=0.7, label="Pearson r" if edge_color_mode=="signed" else "|Pearson r|")
            plt.title(f"{dataset_name} Graph (Pearson-weighted, spring layout)")
            plt.axis("off")
            plt.tight_layout()
            plt.show()

    return data_pyg


# =========================
# Build graphs (Pearson) + visualize
# =========================
# thresholds
clinical_threshold  = 0.30
cna_threshold       = 0.60
gse_threshold       = 0.30
dna_threshold       = 0.30
mirna_threshold     = 0.35
mutation_threshold  = 0.25
coexp_threshold     = 0.40

# neighbors
n_neighbors_clinical = 10
n_neighbors_cna      = 5
n_neighbors_gse      = 5
n_neighbors_dna      = 5
n_neighbors_mirna    = 5
n_neighbors_mutation = 5
n_neighbors_coexp    = 5

# Build from your selected arrays (ensure these exist in scope)
graph_clinical  = construct_knn_graph_pearson(selected_clinical_np,  "Clinical",      n_neighbors_clinical, clinical_threshold,  plot=True, edge_color_mode="signed")
graph_cna       = construct_knn_graph_pearson(selected_cna_np,       "CNA",           n_neighbors_cna,      cna_threshold,       plot=True, edge_color_mode="signed")
graph_gse       = construct_knn_graph_pearson(selected_gse_np,       "GSE",           n_neighbors_gse,      gse_threshold,       plot=True, edge_color_mode="signed")
graph_dna       = construct_knn_graph_pearson(selected_dna_np,       "DNA",           n_neighbors_dna,      dna_threshold,       plot=True, edge_color_mode="signed")
graph_mirna     = construct_knn_graph_pearson(selected_mirna_np,     "miRNA",         n_neighbors_mirna,    mirna_threshold,     plot=True, edge_color_mode="signed")
graph_mutation  = construct_knn_graph_pearson(selected_mutation_np,  "Mutation",      n_neighbors_mutation, mutation_threshold,  plot=True, edge_color_mode="signed")
graph_coexp     = construct_knn_graph_pearson(selected_coexp_np,     "CO-EXPRESSION", n_neighbors_coexp,    coexp_threshold,     plot=True, edge_color_mode="signed")

# Quick shapes
print(f"Clinical: X={graph_clinical.x.shape}, edges={graph_clinical.edge_index.shape}")
print(f"CNA:      X={graph_cna.x.shape}, edges={graph_cna.edge_index.shape}")
print(f"GSE:      X={graph_gse.x.shape}, edges={graph_gse.edge_index.shape}")
print(f"DNA:      X={graph_dna.x.shape}, edges={graph_dna.edge_index.shape}")
print(f"miRNA:    X={graph_mirna.x.shape}, edges={graph_mirna.edge_index.shape}")
print(f"Mutation: X={graph_mutation.x.shape}, edges={graph_mutation.edge_index.shape}")
print(f"COEXP:    X={graph_coexp.x.shape}, edges={graph_coexp.edge_index.shape}")
# ======== CORRECTED: Graph Autoencoder training for feature reconstruction ========
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
from torch_geometric.data import Data

# ---------------- GAE definition ----------------
class GAE_Encoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = pyg_nn.GCNConv(in_channels, 2 * out_channels)
        self.bn1   = nn.BatchNorm1d(2 * out_channels)
        self.conv2 = pyg_nn.GCNConv(2 * out_channels, out_channels)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        return x  # latent

class GraphAutoencoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.encoder = GAE_Encoder(in_channels, out_channels)
        self.decoder = nn.Linear(out_channels, in_channels)  # feature recon

    def forward(self, x, edge_index):
        z = self.encoder(x, edge_index)
        x_hat = self.decoder(z)
        return z, x_hat

# ---------------- Utilities ----------------
def ensure_tensor_features(graph_data: Data):
    """Ensure graph_data.x (float32) and graph_data.edge_index (long) exist and are valid."""
    assert hasattr(graph_data, "x") and graph_data.x is not None, "graph_data.x is missing"
    assert hasattr(graph_data, "edge_index") and graph_data.edge_index is not None, "edge_index is missing"
    graph_data.x = graph_data.x.float()
    graph_data.edge_index = graph_data.edge_index.long()
    return graph_data

def attach_features_if_missing(graph_data: Data, features_np: np.ndarray):
    """If graph_data.x is missing, attach given features."""
    if (not hasattr(graph_data, "x")) or (graph_data.x is None):
        graph_data.x = torch.tensor(features_np, dtype=torch.float32)
    else:
        # If present, ensure dtype float32
        graph_data.x = graph_data.x.float()
    return graph_data

def train_gae(graph_data: Data, out_channels=32, epochs=100, lr=1e-2, weight_decay=0.0, device=None, verbose=True):
    graph_data = ensure_tensor_features(graph_data)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    in_channels = graph_data.x.size(1)  # <--- infer input dim here (FIX)
    model = GraphAutoencoder(in_channels, out_channels).to(device)
    x = graph_data.x.to(device)
    edge_index = graph_data.edge_index.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    model.train()
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        z, x_hat = model(x, edge_index)
        loss = loss_fn(x_hat, x)
        loss.backward()
        optimizer.step()
        if verbose and (epoch % 10 == 0 or epoch == 1 or epoch == epochs):
            print(f"[GAE] Epoch {epoch:3d}/{epochs} | recon MSE: {loss.item():.6f}")

    model.eval()
    with torch.no_grad():
        z, _ = model(x, edge_index)
    return z.cpu().numpy()

# ---------------- OPTIONAL: Attach features to graphs if .x is missing ----------------
# Expect these numpy arrays already exist:
# selected_clinical_np, selected_cna_np, selected_gse_np, selected_mirna_np,
# selected_mutation_np, selected_dna_np, selected_coexp_np
# and graphs: graph_clinical, graph_cna, graph_gse, graph_mirna, graph_mutation, graph_dna, graph_coexp

# If your PyG Data objects don't have .x yet, attach from the selected_* arrays:
graph_clinical = attach_features_if_missing(graph_clinical, selected_clinical_np)
graph_cna      = attach_features_if_missing(graph_cna,      selected_cna_np)
graph_gse      = attach_features_if_missing(graph_gse,      selected_gse_np)
graph_mirna    = attach_features_if_missing(graph_mirna,    selected_mirna_np)
graph_mutation = attach_features_if_missing(graph_mutation, selected_mutation_np)
graph_dna      = attach_features_if_missing(graph_dna,      selected_dna_np)
graph_coexp    = attach_features_if_missing(graph_coexp,    selected_coexp_np)

# Ensure dtypes are correct
for g in [graph_clinical, graph_cna, graph_gse, graph_mirna, graph_mutation, graph_dna, graph_coexp]:
    ensure_tensor_features(g)

# ---------------- Train per-modality GAEs and save ----------------
latent_dim = 32
epochs = 100
lr = 0.01

latent_clinical = train_gae(graph_clinical, out_channels=latent_dim, epochs=epochs, lr=lr)
latent_cna      = train_gae(graph_cna,      out_channels=latent_dim, epochs=epochs, lr=lr)
latent_gse      = train_gae(graph_gse,      out_channels=latent_dim, epochs=epochs, lr=lr)
latent_mirna    = train_gae(graph_mirna,    out_channels=latent_dim, epochs=epochs, lr=lr)
latent_mutation = train_gae(graph_mutation, out_channels=latent_dim, epochs=epochs, lr=lr)
latent_dna      = train_gae(graph_dna,      out_channels=latent_dim, epochs=epochs, lr=lr)
latent_coexp    = train_gae(graph_coexp,    out_channels=latent_dim, epochs=epochs, lr=lr)

print("\nStep 5: Integrating Node Embeddings")
print("====================================")
print(f"Clinical Embeddings Shape: {latent_clinical.shape}")
print(f"CNA Embeddings Shape     : {latent_cna.shape}")
print(f"GSE Embeddings Shape     : {latent_gse.shape}")
print(f"DNA Embeddings Shape     : {latent_dna.shape}")
print(f"Mutation Embeddings Shape: {latent_mutation.shape}")
print(f"miRNA Embeddings Shape   : {latent_mirna.shape}")
print(f"COEXP Embeddings Shape   : {latent_coexp.shape}")

# (Optional) Also print raw feature shapes if you keep the selected_* arrays around
try:
    print(f"\nClinical Shape: {selected_clinical_np.shape}")
    print(f"CNA Shape     : {selected_cna_np.shape}")
    print(f"GSE Shape     : {selected_gse_np.shape}")
    print(f"DNA Shape     : {selected_dna_np.shape}")
    print(f"Mutation Shape: {selected_mutation_np.shape}")
    print(f"miRNA Shape   : {selected_mirna_np.shape}")
    print(f"COEXP Shape   : {selected_coexp_np.shape}")
except NameError:
    pass

# Save outputs
np.save("latent_clinical.npy", latent_clinical)
np.save("latent_cna.npy",      latent_cna)
np.save("latent_gse.npy",      latent_gse)
np.save("latent_dna.npy",      latent_dna)
np.save("latent_mirna.npy",    latent_mirna)
np.save("latent_mutation.npy", latent_mutation)
np.save("latent_coexp.npy",    latent_coexp)
np.save("labels.npy",          np.asarray(labels))
latent_clinical=np.load("latent_clinical.npy")
latent_cna=np.load("latent_cna.npy")
latent_gse=np.load("latent_gse.npy")
latent_dna=np.load("latent_dna.npy")
latent_mirna=np.load("latent_mirna.npy")
latent_mutation=np.load("latent_mutation.npy")
latent_coexp=np.load("latent_coexp.npy")
import numpy as np

def compare_selected_latent(selected, latent, name="modality"):
    """
    Compare a selected features array and its corresponding latent embeddings.
    - Reports shapes
    - Checks sample count match
    - Declares 'different' if column counts differ (expected for embeddings)
    - For a numeric sanity metric, computes mean absolute diff on the first min(cols) columns
    """
    n_sel, d_sel = selected.shape
    n_lat, d_lat = latent.shape

    print(f"\n[{name.upper()}]")
    print(f"  Selected shape: {selected.shape}")
    print(f"  Latent shape:   {latent.shape}")

    # Row count check
    if n_sel == n_lat:
        print("  ✅ Same number of samples (rows)")
    else:
        print("  ❌ Different number of samples!")
        return

    # If column counts differ -> they are different (expected)
    if d_sel != d_lat:
        print("  ✅ Different dimensionality (expected: embeddings compress/transform features)")
        k = min(d_sel, d_lat)
        mad = np.mean(np.abs(selected[:, :k] - latent[:, :k]))
        print(f"  Mean absolute difference on first {k} columns: {mad:.6f}")
        return

    # If same columns, we can test numeric closeness
    if np.allclose(selected, latent, atol=1e-6, rtol=1e-6):
        print("  ⚠️ Arrays are nearly identical (unexpected for learned embeddings)")
    else:
        mad = np.mean(np.abs(selected - latent))
        print("  ✅ Arrays differ (as expected)")
        print(f"  Mean absolute difference (all cols): {mad:.6f}")


# 🔹 Call for all modalities (make sure these variables exist in memory)
compare_selected_latent(selected_clinical_np, latent_clinical, name="clinical")
compare_selected_latent(selected_cna_np,      latent_cna,      name="cna")
compare_selected_latent(selected_gse_np,      latent_gse,      name="gse")
compare_selected_latent(selected_dna_np,      latent_dna,      name="dna")
compare_selected_latent(selected_mirna_np,    latent_mirna,    name="mirna")
compare_selected_latent(selected_mutation_np, latent_mutation, name="mutation")
compare_selected_latent(selected_coexp_np,    latent_coexp,    name="coexp")
import numpy as np

def check_nan(name, arr):
    """Check if array has any NaN values and print results."""
    if arr is None:
        print(f"{name}: [SKIPPED] (array is None)")
        return
    arr = np.asarray(arr)
    has_nan = np.isnan(arr).any()
    n_nans  = np.isnan(arr).sum()
    print(f"{name:<20} | Shape: {arr.shape} | Any NaN: {has_nan} | Total NaNs: {n_nans}")

print("\n=== Checking SELECTED features ===")
check_nan("selected_clinical", selected_clinical_np)
check_nan("selected_cna", selected_cna_np)
check_nan("selected_gse", selected_gse_np)
check_nan("selected_dna", selected_dna_np)
check_nan("selected_mirna", selected_mirna_np)
check_nan("selected_mutation", selected_mutation_np)
check_nan("selected_coexp", selected_coexp_np)

print("\n=== Checking LATENT embeddings ===")
check_nan("latent_clinical", latent_clinical)
check_nan("latent_cna", latent_cna)
check_nan("latent_gse", latent_gse)
check_nan("latent_dna", latent_dna)
check_nan("latent_mirna", latent_mirna)
check_nan("latent_mutation", latent_mutation)
check_nan("latent_coexp", latent_coexp)
------------Combine RAW Feature with Graph Embedding----
import numpy as np
import pandas as pd

# -------------------------------
# Dictionary of latent + raw arrays
# (assumes they are already loaded in memory)
# -------------------------------
modalities = {
    "Clinical":   (latent_clinical, selected_clinical_np),
    "CNA":        (latent_cna, selected_cna_np),
    "GSE":        (latent_gse, selected_gse_np),
    "DNA":        (latent_dna, selected_dna_np),
    "miRNA":      (latent_mirna, selected_mirna_np),
    "Mutation":   (latent_mutation, selected_mutation_np),
    "COEXP":      (latent_coexp, selected_coexp_np),
}

# -------------------------------
# Combine and save for each modality
# -------------------------------
for name, (latent, raw) in modalities.items():
    print(f"\n🔹 Processing {name}...")
    if latent.shape[0] != raw.shape[0]:
        raise ValueError(f"Row mismatch in {name}: latent={latent.shape[0]}, raw={raw.shape[0]}")

    combined = np.concatenate([latent, raw], axis=1)
    print(f"   {name} combined shape: {combined.shape}")

    # Save as .npy
    np.save(f"combined_{name.lower()}.npy", combined)
    # Save as .csv
    pd.DataFrame(combined).to_csv(f"combined_{name.lower()}.csv", index=False)

print("\n✅ All combined files saved (both .npy and .csv).")
--------Classifier------------
combined_clinical=np.load("combined_clinical.npy")
combined_cna=np.load("combined_cna.npy")
combined_gse=np.load("combined_gse.npy")
combined_dna=np.load("combined_dna.npy")
combined_mirna=np.load("combined_mirna.npy")
combined_mutation=np.load("combined_mutation.npy")
combined_coexp=np.load("combined_coexp.npy")






