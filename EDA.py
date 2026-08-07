import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch

# settings
pd.set_option("display.max_columns", None)
sns.set_theme(style="whitegrid")

# load data set
df = pd.read_csv("./star_classification.csv")

# remove unused columns
cols = ["u", "g", "r", "i", "z", "redshift"]
drop_cols = ["obj_ID", "alpha", "delta", "run_ID", "rerun_ID", "cam_col", "field_ID", "spec_obj_ID", "plate", "MJD", "fiber_ID"]
df.drop(columns=drop_cols, inplace=True)

# remove rows containing -9999 invalid measurements
before = len(df)
rows_to_drop = df[(df[cols] == -9999).any(axis=1)]
df = df.drop(index=rows_to_drop.index)
after = len(df)

# remove duplicate rows
duplicates = df.duplicated().sum()
df = df.drop_duplicates().reset_index(drop=True)

print("Invalid rows removed:", before-after)
print("Duplicates removed:", duplicates)
print("Remaining rows:", len(df))

print("\n" + "-" * 70)
print("Exploratory Data Analysis")
print("-" * 70)

print("\nDataset Information")
print(df.info())

print("\nSummary Statistics")
print(df.describe())

print("\nClass Counts")
print(df['class'].value_counts())

print("Generating class distribution...")
sns.countplot(data=df, x="class")
plt.title("Class Distribution")
plt.savefig("./EDA_Charts/class_distribution.png")
plt.close()

print("Generating redshift boxplot...")
plt.figure(figsize=(12,6))
df[["redshift"]].boxplot()
plt.title("Boxplot of Redshift")
plt.savefig("./EDA_Charts/redshift_boxplot.png")
plt.close()

print("Generating other boxplots...")
plt.figure(figsize=(8, 6))
df[["u", "g", "r", "i", "z"]].boxplot()
plt.title("Boxplots of Key Features")
plt.xticks(rotation=45)
plt.savefig("./EDA_Charts/features_boxplot.png")
plt.close()

print("Generating Correlation Heatmap")

plt.figure(figsize=(12,10))
sns.heatmap(
    df.corr(numeric_only=True),
    cmap="coolwarm",
    annot=False
)
plt.title("Correlation Heatmap")
plt.savefig("./EDA_Charts/heatmap_correlation.png")
plt.close()

print("Generating class-versus-feature boxplots...")
for feature in cols:
    plt.figure(figsize=(9, 6))
    sns.boxplot(
        data=df,
        x="class",
        y=feature,
        showfliers=False,
    )

    plt.title(f"{feature.capitalize()} Distribution by Class")
    plt.xlabel("Class")
    plt.ylabel(feature)
    plt.tight_layout()
    plt.savefig(
        f"./EDA_Charts/{feature}_comparison_boxplot.png",
        dpi=300,
    )
    plt.close()

sample_df = df.sample(n=5000, random_state=42)
print("Genearting Feature Comparisons...")
plt.figure(figsize=(8,6))
sns.scatterplot(
    data=sample_df,
    x="redshift",
    y="u",
    hue="class",
    alpha=0.5
)
plt.title("Redshift vs u")
plt.tight_layout()
plt.savefig("./EDA_Charts/redshift_vs_u.png")
plt.close()

plt.figure(figsize=(8,6))
sns.scatterplot(
    data=sample_df,
    x="redshift",
    y="g",
    hue="class",
    alpha=0.5
)
plt.title("Redshift vs g")
plt.tight_layout()
plt.savefig("./EDA_Charts/redshift_vs_g.png")
plt.close()

plt.figure(figsize=(8,6))
sns.scatterplot(
    data=sample_df,
    x="u",
    y="z",
    hue="class",
    alpha=0.5
)
plt.title("u vs z")
plt.tight_layout()
plt.savefig("./EDA_Charts/u_vs_z.png")
plt.close()

plt.figure(figsize=(8,6))
sns.scatterplot(
    data=sample_df,
    x="g",
    y="r",
    hue="class",
    alpha=0.5
)
plt.title("g vs r")
plt.tight_layout()
plt.savefig("./EDA_Charts/g_vs_r.png")
plt.close()

plt.figure(figsize=(8,6))
sns.scatterplot(
    data=sample_df,
    x="r",
    y="i",
    hue="class",
    alpha=0.5
)
plt.title("r vs i")
plt.tight_layout()
plt.savefig("./EDA_Charts/r_vs_i.png")
plt.close()
print(f"\nAll charts saved to EDA_Charts")