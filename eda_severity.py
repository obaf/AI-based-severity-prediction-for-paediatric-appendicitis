import pandas as pd
import numpy as np

df = pd.read_csv("appendicitis_dataset.csv")
print("FULL SHAPE:", df.shape)

# Target: Severity
print("\n--- Severity value counts (raw) ---")
print(df["Severity"].value_counts(dropna=False))

print("\n--- Diagnosis value counts ---")
print(df["Diagnosis"].value_counts(dropna=False))

print("\n--- Management value counts ---")
print(df["Management"].value_counts(dropna=False))

# Rows usable for severity (non-null severity)
sev = df[df["Severity"].notna()].copy()
print("\nRows with non-null Severity:", sev.shape[0])
print(sev["Severity"].value_counts(normalize=True).round(3))

# Missingness per column (top)
print("\n--- Missingness fraction (sorted desc), among severity-labeled rows ---")
miss = sev.isna().mean().sort_values(ascending=False)
print(miss.round(3).to_string())
