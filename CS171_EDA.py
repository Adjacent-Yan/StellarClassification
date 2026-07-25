import pandas as pd
pd.set_option('display.max_columns', None)
df = pd.read_csv("../star_classification.csv")
drop_cols = ["obj_ID", "alpha", "delta", "run_ID", "rerun_ID", "cam_col", "field_ID", "spec_obj_ID", "plate", "MJD", "fiber_ID"]
df.drop(columns=drop_cols, inplace=True)
cols = ["u", "g", "r", "i", "z"]
before = len(df)
rows_to_drop = df[(df[cols] == -9999).any(axis=1)]
df = df.drop(index=rows_to_drop.index)
after = len(df)

print("\n" + "-" * 70)
print("Exploratory Data Analysis")
print("-" * 70)
print(f"Outliers dropped: {before - after}")
print("Number of duplicates:", df.duplicated().sum())
print()
print(df.info())
print()
print(df.describe())
print()
print(df['class'].value_counts())
