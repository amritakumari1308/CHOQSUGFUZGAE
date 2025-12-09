## Hyperparameter Configuration
All hyperparameter settings used in theFuMOGAE experiments are stored in:
GAE:
  encoder_layers: [256, 128]
  activation: "ReLU"
  dropout: 0.40
  learning_rate: 0.010
  epochs: 10

GraphConstruction:
  similarity_metric: "Pearson"
  kNN:
    clinical: 10
    others: 5
  thresholds:
    CLN: 0.30
    CNA: 0.60
    GSE: 0.30
    DNA: 0.30
    miRNA: 0.35
    MUT: 0.25
    COEXP: 0.40

Classifier_FFNN:
  hidden_sizes: [256, 128, 64]
  dropout: 0.40
  optimizer: "AdamW"
  learning_rate: 0.001
  epochs: 200
  batch_size: 128

OtherClassifiers:
  RF:
    trees: 400
    class_weight: "balanced_subsample"
  XGBoost:
    trees: 600
    depth: 6
    eta: 0.03
    subsample: 0.9
    colsample: 0.8
  SVM:
    kernel: "RBF"
    C: 2.0
    gamma: "scale"
    class_weight: "balanced"
  MLP:
    max_iter: 200
    lr_schedule: "adaptive"

Training:
  cross_validation:
    outer: 10
    inner: 5
  imbalance: "SMOTE"

FuzzyFusion:
  alpha_values: [0.3, 0.5, 0.7, 0.9]
  formula: "alpha*Choquet + (1-alpha)*Sugeno"
