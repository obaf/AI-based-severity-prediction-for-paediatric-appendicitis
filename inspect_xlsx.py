import pandas as pd

xls = pd.ExcelFile("app_data.xlsx")
print("SHEETS:", xls.sheet_names)

df = pd.read_excel("app_data.xlsx", sheet_name=0)
print("\nShape:", df.shape)
print("\nColumns (first 15):", list(df.columns[:15]))

# US_Number and Severity
for col in ["US_Number", "Severity", "Diagnosis", "Management"]:
    if col in df.columns:
        print(f"\n--- {col} ---")
        print(df[col].value_counts(dropna=False).head(8))

print("\nUS_Number dtype:", df["US_Number"].dtype if "US_Number" in df.columns else "MISSING")
print("US_Number sample:", df["US_Number"].dropna().head(10).tolist() if "US_Number" in df.columns else "")
print("Rows with US_Number not null:", df["US_Number"].notna().sum() if "US_Number" in df.columns else 0)
print("Rows with Severity not null:", df["Severity"].notna().sum() if "Severity" in df.columns else 0)

# Test codes
test_codes = pd.read_csv("test_set_codes.csv", header=None)[0].tolist()
print("\nN test codes:", len(test_codes), "sample:", test_codes[:10])
overlap = df[df["US_Number"].isin(test_codes)]
print("Test codes matched in US_Number:", len(overlap))
print("Test set severity distribution:")
print(overlap["Severity"].value_counts(dropna=False))
