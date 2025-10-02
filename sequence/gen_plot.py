import pandas as pd
import matplotlib.pyplot as plt

# Read metrics CSV
df = pd.read_csv("D:/OneDrive/online_research/Sep/Breast_cancer_TMS-CMAN/sequence/logs/3/train_metrics.csv")

# Filter out non-numeric epochs (e.g., final_test)
df = df[pd.to_numeric(df["epoch"], errors="coerce").notnull()]
df["epoch"] = df["epoch"].astype(int)

# Extract series used for plotting
epochs = df["epoch"]
train_acc = df["train_acc"]
val_acc = df["val_acc"]
val_auc = df["val_auc"]

# Create figure
plt.figure(figsize=(12, 6))

# Subplot 1: Accuracy
plt.subplot(1, 2, 1)
plt.plot(epochs, train_acc, label="Train Accuracy", marker='o')
plt.plot(epochs, val_acc, label="Validation Accuracy", marker='o')
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Train & Validation Accuracy")
plt.legend()
plt.grid(True)

# Subplot 2: AUC
plt.subplot(1, 2, 2)
plt.plot(epochs, val_auc, label="Validation AUC", color='green', marker='o')
plt.xlabel("Epoch")
plt.ylabel("AUC")
plt.title("Validation AUC")
plt.legend()
plt.grid(True)

# Render figure
plt.tight_layout()
plt.show()