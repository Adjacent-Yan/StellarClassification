import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch
pd.set_option('display.max_columns', None)
df = pd.read_csv("./star_classification.csv")
drop_cols = ["obj_ID", "alpha", "delta", "run_ID", "rerun_ID", "cam_col", "field_ID", "spec_obj_ID", "plate", "MJD", "fiber_ID"]
df.drop(columns=drop_cols, inplace=True)
cols = ["u", "g", "r", "i", "z", "redshift"]
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

print("Generating Class Distribution")
sns.countplot(data=df, x="class")
plt.title("Class Distribution")
plt.savefig("./EDA_Charts/class_distribution.png")

numeric_cols = df.select_dtypes(include="number").columns

print("Generating box plots")
plt.figure(figsize=(12,6))
df[["redshift"]].boxplot()
plt.title("Boxplot of Redshift")
plt.savefig("./EDA_Charts/redshift_boxplot.png")
plt.figure(figsize=(8, 6))
df[["u", "g", "r", "i", "z"]].boxplot()
plt.title("Boxplots of Key Features")
plt.xticks(rotation=45)
plt.savefig("./EDA_Charts/features_boxplot.png")

plt.figure(figsize=(12,10))

print("Generating Correlation Heatmap")
sns.heatmap(
    df.corr(numeric_only=True),
    cmap="coolwarm",
    annot=False
)

plt.title("Correlation Heatmap")
plt.savefig("./EDA_Charts/heatmap_correlation.png")