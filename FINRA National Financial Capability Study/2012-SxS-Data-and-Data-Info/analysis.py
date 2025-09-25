#How does financial literacy vary across age groups and genders in the 2012 NFCS data?
#How does financial literacy vary across regions and divisions in the 2012 NFCS data?

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# === 1) Load cleaned dataset ===
df = pd.read_csv("cleaned_NFCS_2012.csv")

print(df.head())
print(df.info())


# ============== 2) Build financial literacy score (0–5) ==============
LITERACY_COLS = ["Q_Interest", "Q_Inflation", "Q_Bonds", "Q_Risk", "Q_Mortgage"]
correct_answers = {
    "Q_Interest": "More than $102",
    "Q_Inflation": "More than today",
    "Q_Bonds": "Fall",
    "Q_Risk": "True",
    "Q_Mortgage": "False",
}
for col in LITERACY_COLS:
    df[col + "_score"] = (df[col] == correct_answers[col]).astype(int)

df["literacy_score"] = df[[c + "_score" for c in LITERACY_COLS]].sum(axis=1)

# ============== 3) Create age groups ==============
if "Age" in df.columns:
    # If you have numeric Age after cleaning, bin it; otherwise fall back to Age_Category_Code
    df["Age_numeric"] = pd.to_numeric(df["Age"], errors="coerce")
    bins = [17, 24, 34, 44, 54, 64, 120]
    labels = ["18-24", "25-34", "35-44", "45-54", "55-64", "65+"]
    df["Age_group"] = pd.cut(df["Age_numeric"], bins=bins, labels=labels, right=True)
elif "Age_Category_Code" in df.columns:
    # NFCS often encodes age as ordered categories already (1..6). Keep it categorical labels for readability.
    code_to_label = {1:"18-24", 2:"25-34", 3:"35-44", 4:"45-54", 5:"55-64", 6:"65+"}
    df["Age_group"] = df["Age_Category_Code"].map(code_to_label).fillna(df["Age_Category_Code"])
    
# ============== 4) Helpers: unweighted mean / SE only ==============
import numpy as np
import pandas as pd

def summarize(group_cols):
    g = df.groupby(group_cols)["literacy_score"].agg(["count", "mean", "std"]).reset_index()
    g = g.rename(columns={"count": "n"})
    g["se"] = g["std"] / np.sqrt(g["n"].clip(lower=1))
    return g

# ============== 5) Tables ==============
age_tbl = summarize(["Age_group"])
gender_tbl = summarize(["Gender"])
age_gender_tbl = summarize(["Age_group", "Gender"])

print("\n=== Literacy by Age Group ===")
print(age_tbl)
print("\n=== Literacy by Gender ===")
print(gender_tbl)
print("\n=== Literacy by Age × Gender ===")
print(age_gender_tbl)

# ============== 6) Plots ==============
plt.rcParams["figure.dpi"] = 140

# Plot 1: Age 
plt.figure(figsize=(8,6))
x = age_tbl["Age_group"].astype(str)
y = age_tbl["mean"]
yerr = age_tbl["se"]
plt.errorbar(x, y, yerr=yerr, fmt='-o', capsize=4)
plt.xlabel("Age Group")
plt.ylabel("Average Literacy Score (0–5)")
plt.title("Financial Literacy by Age Group (NFCS 2012)")
plt.tight_layout()
plt.savefig("literacy_by_age.png")


# Plot 2: Gender (bar chart with custom colors)
plt.figure(figsize=(6,6))
x = gender_tbl["Gender"].astype(str)
y = gender_tbl["mean"]
yerr = gender_tbl["se"]

colors = ["pink" if g == "Female" else "skyblue" for g in x]

plt.bar(x, y, yerr=yerr, capsize=4, color=colors)
plt.xlabel("Gender")
plt.ylabel("Average Literacy Score (0–5)")
plt.title("Financial Literacy by Gender (NFCS 2012)")
plt.tight_layout()
plt.savefig("literacy_by_gender.png")



# Plot 3: Age × Gender (interaction)
plt.figure(figsize=(9,6))

color_map = {"Female": "pink", "Male": "skyblue"}

for g in age_gender_tbl["Gender"].dropna().unique():
    sub = age_gender_tbl[age_gender_tbl["Gender"] == g]
    plt.plot(
        sub["Age_group"].astype(str),
        sub["mean"],
        marker="o",
        label=str(g),
        color=color_map.get(g, "gray")
    )

plt.xlabel("Age Group")
plt.ylabel("Average Literacy Score (0–5)")
plt.title("Financial Literacy by Age and Gender (NFCS 2012)")
plt.legend(title="Gender")
plt.tight_layout()
plt.savefig("literacy_by_age_gender.png")

# Plot 4: Region (Census_Region) 
# Helper to add value labels on top of bars
def _add_value_labels(ax, decimals=2):
    for p in ax.patches:
        height = p.get_height()
        ax.annotate(f"{height:.{decimals}f}",
                    (p.get_x() + p.get_width()/2, height),
                    ha="center", va="bottom", fontsize=9, xytext=(0, 3),
                    textcoords="offset points")

SHOW_VALUE_LABELS = True  # set to False if you don't want numbers on bars

# Optional readable labels for US Census Regions
_region_labels = {1: "Northeast", 2: "Midwest", 3: "South", 4: "West"}

if "Census_Region" in df.columns:
    # Summarize
    region_tbl = summarize(["Census_Region"])

    # Order by numeric code if possible
    region_tbl["_order"] = pd.to_numeric(region_tbl["Census_Region"], errors="coerce")
    region_tbl["Region_Label"] = (
        region_tbl["Census_Region"].map(_region_labels)
        .fillna(region_tbl["Census_Region"].astype(str))
    )
    region_tbl = region_tbl.sort_values(by=["_order", "Region_Label"], na_position="last")

    print("\n=== Literacy by Region ===")
    print(region_tbl[["Region_Label", "n", "mean", "std", "se"]])

    # Plot
    import matplotlib.pyplot as plt
    plt.figure(figsize=(9, 6))
    ax = plt.gca()
    x = region_tbl["Region_Label"]
    y = region_tbl["mean"]
    yerr = region_tbl["se"]
    ax.bar(x, y, yerr=yerr, capsize=4, color="mediumseagreen")
    ax.set_xlabel("Region")
    ax.set_ylabel("Average Literacy Score (0–5)")
    ax.set_title("Financial Literacy by Region (NFCS 2012)")
    plt.xticks(rotation=15)
    plt.tight_layout()
    if SHOW_VALUE_LABELS:
        _add_value_labels(ax, decimals=2)
    plt.savefig("literacy_by_region.png")
else:
    print("Column 'Census_Region' not found in dataset.")


# Plot 5: Division (Census_Division) 
# Optional readable labels for US Census Divisions
_division_labels = {
    1: "New England",
    2: "Middle Atlantic",
    3: "East North Central",
    4: "West North Central",
    5: "South Atlantic",
    6: "East South Central",
    7: "West South Central",
    8: "Mountain",
    9: "Pacific",
}

if "Census_Division" in df.columns:
    division_tbl = summarize(["Census_Division"])

    # Keep order by numeric code; make readable labels
    division_tbl["_order"] = pd.to_numeric(division_tbl["Census_Division"], errors="coerce")
    division_tbl["Division_Label"] = (
        division_tbl["Census_Division"].map(_division_labels)
        .fillna(division_tbl["Census_Division"].astype(str))
    )
    division_tbl = division_tbl.sort_values(by=["_order", "Division_Label"], na_position="last")

    print("\n=== Literacy by Division ===")
    print(division_tbl[["Division_Label", "n", "mean", "std", "se"]])

    # Plot
    plt.figure(figsize=(12, 6))  # wider canvas for 9 labels
    ax = plt.gca()
    x = division_tbl["Division_Label"]
    y = division_tbl["mean"]
    yerr = division_tbl["se"]
    ax.bar(x, y, yerr=yerr, capsize=4, color="teal")
    ax.set_xlabel("Division")
    ax.set_ylabel("Average Literacy Score (0–5)")
    ax.set_title("Financial Literacy by Census Division (NFCS 2012)")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    if SHOW_VALUE_LABELS:
        _add_value_labels(ax, decimals=2)
    plt.savefig("literacy_by_division.png")
else:
    print("Column 'Census_Division' not found in dataset.")

#Plot5 Boxplot: Literacy by Region 

plt.figure(figsize=(8,6))
sns.boxplot(x="Census_Region", y="literacy_score", data=df, palette="Set2")
plt.xlabel("Region")
plt.ylabel("Financial Literacy Score (0–5)")
plt.title("Financial Literacy Distribution by Region (NFCS 2012)")
plt.xticks([0,1,2,3], ["Northeast", "Midwest", "South", "West"])
plt.tight_layout()
plt.savefig("literacy_boxplot_region.png")
plt.show()

# Plot6 Boxplot: Literacy by Division
plt.figure(figsize=(11,6))
sns.boxplot(x="Census_Division", y="literacy_score", data=df, palette="Set3")
plt.xlabel("Division")
plt.ylabel("Financial Literacy Score (0–5)")
plt.title("Financial Literacy Distribution by Census Division (NFCS 2012)")
plt.xticks(
    ticks=range(9),
    labels=[
        "New England", "Middle Atlantic", "East North Central", "West North Central",
        "South Atlantic", "East South Central", "West South Central",
        "Mountain", "Pacific"
    ],
    rotation=25,
    ha="right"
)
plt.tight_layout()
plt.savefig("literacy_boxplot_division.png")


    
# ==============Conclusion==============

##Question 1:How does financial literacy vary across age groups and genders in the 2012 NFCS data?
#Conclusion:
#The 2012 NFCS results reveal significant differences in financial literacy across age groups and genders.
#Age Effect
#Financial literacy scores rise steadily with age. Young adults (18–24) display the lowest average knowledge, while respondents aged 65 and older achieve the highest scores. This pattern indicates that financial knowledge accumulates through life experience, exposure to financial products, and long-term decision-making. However, it also raises concerns that younger populations may lack the literacy needed for early financial choices such as student loans, credit cards, or retirement planning.

#Gender Effect
#Men consistently outperform women in financial literacy, with an average gap of roughly half a point on the 5-question index. This disparity aligns with previous research showing differences in confidence, financial education, and opportunities to engage with investment or household financial decisions.

#Age × Gender Interaction
#Across all age groups, men maintain higher scores than women. Both genders improve with age, but the gap remains persistent rather than narrowing. This suggests structural or cultural factors that continue to limit women’s access to financial knowledge, even later in life.

#Interpretation:
#These findings point to the need for targeted financial education, particularly for young adults entering the workforce and for women across all life stages. Tailored interventions could help close the literacy gap and ensure equitable access to financial knowledge that supports better decision-making.


##Question 2: How does financial literacy vary across regions and divisions in the 2012 NFCS data?
#Conclusion:
#The 2012 NFCS data shows that regional differences in financial literacy are modest overall, but some clear patterns emerge:

#Regions: Across the four Census regions, the West has the highest average literacy scores (~2.51), while the South lags slightly behind (~2.36). The Northeast and Midwest fall in between, both averaging around 2.43. This indicates that broad regional variation exists but is relatively small.

#Divisions: At the more detailed division level, differences become more pronounced. Mountain and Pacific divisions lead with the highest average scores (~2.50–2.52), while Middle Atlantic and East South Central divisions record the lowest (~2.29–2.30). Other divisions cluster in between, suggesting that division-level context (such as local education systems or economic conditions) may have a stronger impact than broad regional groupings.

#Distributions: Boxplots reveal wide variation in literacy scores within each region and division, with substantial overlap. This means that while averages differ slightly across geography, individual financial literacy levels vary considerably within each group.

#In summary: Geographic factors do play a role, with the West and certain divisions (Mountain, Pacific) performing slightly better, but the differences are modest compared to the large variation within each region/division.