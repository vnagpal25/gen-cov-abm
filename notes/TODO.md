##
- generate mutations only for S and N proteins, ORF proteins have weird mutation annotation (-3677T)
- construct ancestral tree for Massachusetts
- per protein embeddings using ESM-2 (S, N, orfp, etc..)
- two options to aggregate embeddings per genome
    - attention -> weighted sum
    - MLP all of them -> avg.
- for both, we need a training objective
- training objective could be contrastive loss on predicting parent child relationships (binary classification) -> pushes linked nodes closer together
- alternative training objective contrastive loss on predicting shortest path length