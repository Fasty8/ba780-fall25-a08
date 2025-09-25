# clean_data.py
import pandas as pd
import numpy as np
from pathlib import Path

# ---------- 0) Paths ----------
ROOT = Path(__file__).resolve().parent
SRC_CSV = ROOT / "NFCS 2012 State Data 130503.csv"   # adjust if your CSV lives elsewhere

# ---------- 1) Read & normalize columns ----------
df = pd.read_csv(SRC_CSV)

# Turn pure empty strings into NaN so mapping isn’t confused by blanks
df = df.replace(r"^\s*$", np.nan, regex=True)

# Normalize column names: trim & remove inner spaces
df.columns = df.columns.str.strip().str.replace(r"\s+", "", regex=True)

# Harmonize “2012” suffix & uppercase variants to the keys we use in maps
pre_rename = {
    "A5_2012": "A5",
    "C1_2012": "C1", "C2_2012": "C2", "C3_2012": "C3", "C4_2012": "C4",
    "C5_2012": "C5", "C10_2012": "C10", "C11_2012": "C11",
    "E5_2012": "E5",
    "EA_1": "Ea_1",
    "EA_2A": "Ea_2a",
    "E4A": "E4a",
}
df = df.rename(columns={k: v for k, v in pre_rename.items() if k in df.columns})

# ---------- 2) Dictionaries ----------
DK_REF = {98: "Don't know", 99: "Prefer not to say"}
YES_NO = {1: "Yes", 2: "No", **DK_REF}

FREQ_1_3 = {1: "Frequently", 2: "Sometimes", 3: "Never", **DK_REF}
AGREE_1_7 = {1:"Strongly disagree", 2:"", 3:"", 4:"Neither", 5:"", 6:"", 7:"Strongly agree", **DK_REF}
SCALE_1_7_KNOW = {1:"Very low", 2:"", 3:"", 4:"", 5:"", 6:"", 7:"Very high", **DK_REF}
SCALE_1_10 = {**{i: str(i) for i in range(1, 11)}, **DK_REF}
TRUE_FALSE = {1: "True", 2: "False", **DK_REF}

A3_gender = {1: "Male", 2: "Female"}
A5_edu = {
    1: "Did not complete high school", 2: "High school diploma", 3: "GED/alternative credential",
    4: "Some college", 5: "College graduate", 6: "Post graduate", 99: "Prefer not to say"
}
A6_marital = {1:"Married", 2:"Single", 3:"Separated", 4:"Divorced", 5:"Widowed", 99:"Prefer not to say"}
A7_living = {
    1:"Only adult in household", 2:"Live with spouse/partner", 3:"Live in parents' home",
    4:"Live with other family/friends/roommates", 99:"Prefer not to say"
}
A11_children = {1:"1",2:"2",3:"3",4:"4 or more",5:"No financially dependent children",6:"No children",99:"Prefer not to say"}
A8_income = {
    1:"< $15,000", 2:"$15k–$24,999", 3:"$25k–$34,999", 4:"$35k–$49,999",
    5:"$50k–$74,999", 6:"$75k–$99,999", 7:"$100k–$149,999", 8:"$150k+", **DK_REF
}
AM21_22_service = {1:"Currently member", 2:"Previously member", 3:"Never member", 99:"Prefer not to say"}
A9_work = {
    1:"Self-employed", 2:"Employed full-time", 3:"Employed part-time", 4:"Homemaker",
    5:"Full-time student", 6:"Unable to work", 7:"Unemployed/laid off", 8:"Retired", 99:"Prefer not to say"
}
E4a_buy_year = {
    1:"1999 or earlier", 2:"2000", 3:"2001", 4:"2002", 5:"2003", 6:"2004", 7:"2005",
    8:"2006", 9:"2007", 10:"2008", 11:"2009", 12:"2010", 13:"2011", 14:"2012",
    97:"Did not purchase", **DK_REF
}
F1_cc_num = {1:"1",2:"2–3",3:"4–8",4:"9–12",5:"13–20",6:">20",7:"No credit cards",**DK_REF}
G25_freq = {1:"Never",2:"1 time",3:"2 times",4:"3 times",5:"4+ times",**DK_REF}

# ---------- 3) Column→dictionary registry & prefix groups ----------
COLUMN_MAPS = {
    # Demographics
    "A3": A3_gender, "A5": A5_edu, "A6": A6_marital, "A7": A7_living, "A11": A11_children,
    "A8": A8_income, "AM21": AM21_22_service, "AM22": AM21_22_service, "A9": A9_work,

    # Home & mortgages
    "E4a": E4a_buy_year,

    # Credit cards
    "F1": F1_cc_num,

    # Literacy (often analyzed)
    "M6": {1:"More than $102", 2:"Exactly $102", 3:"Less than $102", **DK_REF},
    "M7": {1:"More than today", 2:"Exactly the same", 3:"Less than today", **DK_REF},
    "M8": {1:"Rise", 2:"Fall", 3:"Same", 4:"No relationship", **DK_REF},
    "M9": TRUE_FALSE, "M10": TRUE_FALSE,

    # Attitudes
    "G23": AGREE_1_7,

    # Scales
    "J1": SCALE_1_10, "J2": SCALE_1_10, "M4": SCALE_1_7_KNOW,
}

PREFIX_MAPS = {
    "B20_": YES_NO, "B22_": FREQ_1_3, "K_": YES_NO, "D20_": YES_NO,
    "F2_": YES_NO, "G25_": G25_freq, "M21_": YES_NO, "M1_": AGREE_1_7,
}

# ---------- 4) Helpers ----------
# Map safely whether values are numbers or strings ("98" vs 98)
def apply_map_safe(s: pd.Series, m: dict) -> pd.Series:
    mixed = {**m, **{str(k): v for k, v in m.items()}}
    return s.map(mixed).fillna(s)

# ---------- 5) Apply mappings ----------
# Specific columns
for col, mapping in COLUMN_MAPS.items():
    if col in df.columns:
        df[col] = apply_map_safe(df[col], mapping)

# Prefix-based batches
for prefix, mapping in PREFIX_MAPS.items():
    for col in [c for c in df.columns if c.startswith(prefix)]:
        df[col] = apply_map_safe(df[col], mapping)

# Special cases
# E5: replace only -98/-99 with text; keep other numeric % values
if "E5" in df.columns:
    df["E5"] = df["E5"].replace({-98:"Don't know",-99:"Prefer not to say"}).replace({"-98":"Don't know","-99":"Prefer not to say"})

# A3a (age): 999 means prefer not to say
if "A3a" in df.columns:
    df["A3a"] = df["A3a"].replace({999:"Prefer not to say", "999":"Prefer not to say"})

# Quick probe to confirm mapping worked
for probe in ["A3","A5","A6","A7","A8","E4a","F1","M6","M7","M8","M9","M10","G23"]:
    if probe in df.columns:
        print(probe, "→ sample:", df[probe].dropna().astype(str).unique()[:6])

# ---------- 6) Light cleanup ----------
# Drop fully empty columns
df = df.dropna(axis=1, how="all")

# Optional: treat DK/Refusal as missing for analysis (uncomment if you want)
# df = df.replace({"Don't know": pd.NA, "Prefer not to say": pd.NA})

# ---------- 6.5) Optional friendly column names ----------
friendly_names = {
    "NFCSID":"Respondent_ID","STATEQ":"State","CENSUSDIV":"Census_Division","CENSUSREG":"Census_Region",
    "A3":"Gender","A3a":"Age","A3Ar_w":"Age_Category_Code","A5":"Education","A6":"Marital_Status",
    "A7":"Living_Arrangement","A8":"Household_Income","AM21":"Military_Service_Self","AM22":"Military_Service_HH",
    "A9":"Employment_Status",
    "wgt_n2":"Weight_N","wgt_d2":"Weight_D","wgt_s3":"Weight_S",
    "M6":"Q_Interest","M7":"Q_Inflation","M8":"Q_Bonds","M9":"Q_Risk","M10":"Q_Mortgage",
}
df = df.rename(columns={k:v for k,v in friendly_names.items() if k in df.columns})

# ---------- 7) Save ----------
out_dir = ROOT / "cleaned_data_2012"
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / "cleaned_NFCS_2012.csv"
df.to_csv(out_path, index=False)
print("WROTE:", out_path.resolve())

# Read back a small sample to verify
df_check = pd.read_csv(out_path, nrows=5)
print("Reloaded columns (first 20):", list(df_check.columns)[:20])
print(df_check.head(3))
