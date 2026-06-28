from ucimlrepo import fetch_ucirepo

# fetch dataset
regensburg_pediatric_appendicitis = fetch_ucirepo(id=938)

# data (as pandas dataframes)
X = regensburg_pediatric_appendicitis.data.features
y = regensburg_pediatric_appendicitis.data.targets

# metadata
print(regensburg_pediatric_appendicitis.metadata)

# variable information
print(regensburg_pediatric_appendicitis.variables)

# save to CSV in the current folder
X.to_csv("appendicitis_features.csv", index=False)
y.to_csv("appendicitis_targets.csv", index=False)

# combined dataset
combined = X.join(y)
combined.to_csv("appendicitis_dataset.csv", index=False)

# variable info for reference
regensburg_pediatric_appendicitis.variables.to_csv("appendicitis_variables.csv", index=False)

print("\nSaved files:")
print("  appendicitis_features.csv  ->", X.shape)
print("  appendicitis_targets.csv   ->", y.shape)
print("  appendicitis_dataset.csv   ->", combined.shape)
print("  appendicitis_variables.csv")
