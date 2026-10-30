import re
import sqlite3
import unicodedata

import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

DB_PATH = "wine_reviews.db"
STYLE_GROUPS_PATH = "wine_style_groups_final.csv"
VARIETY_MAP_PATH = "variety_map.csv"
OUTPUT_PATH = "processed_reviews.pkl"

MIN_CLASS_SIZE = 500
SEED = 67
SPLIT_FRACTIONS = {"train": 0.70, "val": 0.15, "test": 0.15}

NICKNAMES = ["Cab", "Cab Franc", "Cab Sauv", "Zin", "Chard", "Sauv", "Sauv Blanc",
             "Gewürz", "Grüner", "Veltliner", "Pinot", "Cabernet", "Chenin", "Petite"]
NOT_MASKED = {"Melon"}
MASK_TOKEN = "varietyname"

for pkg in ["punkt_tab", "stopwords", "wo9rdnet"]:
    nltk.download(pkg, quiet=True) 

LEMMATIZER = WordNetLemmatizer()

# common words that are used and commonly shared between each group that we found in our data analysis notebook
WINE_WORDS = {"wine", "flavor", "aroma", "palate", "drink", "note", "show",
              "nose", "offer", "give", "come", "year", "hint", "touch", "bit",
              "well", "also", "like", "character", "mouth", "feel", "vineyard"}

# these are words that are within the style names that we want to avoid getting rid of
KEEP = {"red", "white", "apple", "blend", "style", "other", "de", "di", "del", "la",
        "black", "orange", "melon", "vital", "grain", "diamond", "mission", "du"}

def load_reviews(db_path=DB_PATH):
    """
    loads the data base and joins them into one dataframe
    """
    conn = sqlite3.connect(db_path)
    df = pd.read_sql("""
    SELECT r.review_id, r.source,
        l.country, l.province, l.region_1, l.region_2,
        w.winery, v.variety, r.designation, r.title,
        r.points, r.price, r.description,
        t.taster_name, t.twitter_handle AS taster_twitter_handle
    FROM reviews  as r
    LEFT JOIN wineries as w ON r.winery_id = w.winery_id
    LEFT JOIN varieties as v ON r.variety_id = v.variety_id
    LEFT JOIN locations as l ON r.location_id = l.location_id
    LEFT JOIN tasters as t ON r.taster_id = t.taster_id
    ORDER BY r.review_id
    """, conn)
    conn.close()
    return df

def load_style_map(path=STYLE_GROUPS_PATH):
    """
    Will return a map of each style and which varieties belong in each style_group type
    """
    groups = pd.read_csv(path)
    groups["variety"] = groups["varieties"].str.split("; ")
    style_map = groups.explode("variety")[["variety", "style_group", "batch"]] # explodes distributes the list into variety and creates new columns 
    style_map["variety"] = style_map["variety"].str.strip()
    return style_map.reset_index(drop=True)

def add_style_groups(reviews, style_map):
    "adding the new column: style_group (left join keeps every review, like the team file)"
    df = reviews.merge(style_map, on="variety", how="left")
    df["batch"] = df["batch"].astype("Int64")
    print(f"{df['style_group'].notna().sum():,} of {len(df):,} reviews have a style group")
    return df

def add_grape_target(df, path=VARIETY_MAP_PATH):
    """Merge synonyms into one `grape` label, plus color/category/blend info."""
    variety_map = pd.read_csv(path).drop(columns="n_reviews")

    unmapped = set(df["variety"].dropna()) - set(variety_map["variety"])
    assert not unmapped, f"Add these to variety_map.csv: {sorted(unmapped)}"

    df = df.merge(variety_map, on="variety", how="left")
    df[["is_blend", "reviewed"]] = df[["is_blend", "reviewed"]].fillna(False).astype(bool)
    print(f"{df['grape'].notna().sum():,} of {len(df):,} reviews have a grape label")
    return df, variety_map


def mark_target_classes(df, min_size=MIN_CLASS_SIZE):
    """in_target = True for grapes with at least `min_size` reviews."""
    class_sizes = df["grape"].value_counts()
    target_grapes = class_sizes[class_sizes >= min_size].index.tolist()
    df["in_target"] = df["grape"].isin(target_grapes)
    print(f"{len(target_grapes)} classes with >= {min_size} reviews; "
          f"{df['in_target'].sum():,} reviews in a target class")
    return df

def strip_accents(text):
    return "".join(ch for ch in unicodedata.normalize("NFKD", text)
                   if not unicodedata.combining(ch))


def mask_grape_names(df, variety_map):
    """description_masked: every grape name, synonym and nickname -> MASK_TOKEN."""
    checked = variety_map[variety_map["label_type"].isin(["grape", "ambiguous"])]
    names = set(checked["variety"]) | set(checked["grape"].dropna()) | set(NICKNAMES)
    names = {n for n in names if n not in NOT_MASKED}
    names |= {strip_accents(n) for n in names}

    # longest first, so "Cabernet Sauvignon" is replaced before "Cabernet"
    terms = sorted(names, key=len, reverse=True)
    pattern = re.compile(r"\b(?:" + "|".join(re.escape(t) for t in terms) + r")s?\b",
                         flags=re.IGNORECASE)

    df["description_masked"] = df["description"].str.replace(pattern, MASK_TOKEN, regex=True)
    changed = df["description"].ne(df["description_masked"]).mean()
    print(f"{len(terms):,} names masked; {changed:.1%} of reviews changed")
    return df

def add_split(df, seed=SEED):
    """Fixed 70/15/15 split, stratified by grape, identical to the team notebook."""
    rng = np.random.default_rng(seed)
    df["split"] = ""
    for _, idx in df.groupby(df["grape"].fillna("(none)")).groups.items():
        idx = rng.permutation(idx)
        n_train = round(len(idx) * SPLIT_FRACTIONS["train"])
        n_val = round(len(idx) * SPLIT_FRACTIONS["val"])
        df.loc[idx[:n_train], "split"] = "train"
        df.loc[idx[n_train:n_train + n_val], "split"] = "val"
        df.loc[idx[n_train + n_val:], "split"] = "test"
    print(df["split"].value_counts().to_string())
    return df


# ------- Applying the Text Cleaning -------
# Essentially all of these are fixing up all of the tokenization and descriptions similar to how we did it in data analysis
def base_tokens(text):
    return [LEMMATIZER.lemmatize(t) for t in word_tokenize(text.lower()) if t.isalpha()]

def name_words(names):
    """Turn variety names into the same token form as the reviews."""
    return {t for name in names for t in base_tokens(name)}

def build_stopwords(style_map):
    """English stopwords + wine filler + every variety-name word (minus KEEP)."""
    english = set(stopwords.words("english"))
    english |= {LEMMATIZER.lemmatize(w) for w in english}   # catches has -> ha, was -> wa, does -> doe
    variety_words = name_words(style_map["variety"]) - KEEP
    return english | WINE_WORDS | variety_words



def remove_stops(tokens, stops):
    return [t for t in tokens if t not in stops]

def leakage_report(df, style_map, col):
    """% of each style group's reviews that mention one of its OWN variety words....."""
    rates = {}
    for style, varieties in style_map.groupby("style_group")["variety"]:
        own = name_words(varieties) - KEEP
        sub = df[df["style_group"] == style]
        rates[style] = round(100 * sub[col].apply(lambda toks: bool(own & set(toks))).mean(), 1)
    return pd.Series(rates, name=col)

def process():
    reviews = load_reviews()
    style_map = load_style_map()
    df = add_style_groups(reviews, style_map)
    df, variety_map = add_grape_target(df)
    df = mark_target_classes(df)
    df = mask_grape_names(df, variety_map)
    df["word_count"] = df["description"].str.split().str.len()
    df = add_split(df)

    english = set(stopwords.words("english"))
    all_stops = build_stopwords(style_map)
    print(f"Stopwords: {len(all_stops):,} total")

    print("Tokenizing (takes a few minutes)...")
    base = df["description"].apply(base_tokens)
    df["tokens_raw"] = base.apply(remove_stops, stops=english)
    df["tokens"] = base.apply(remove_stops, stops=all_stops)
    df["clean_text"] = df["tokens"].str.join(" ")
    df["clean_designation"] = (df["designation"].fillna("")
                                 .apply(base_tokens)
                                 .apply(remove_stops, stops=all_stops)
                                 .str.join(" "))

    report = pd.concat([leakage_report(df, style_map, "tokens_raw"),
                        leakage_report(df, style_map, "tokens")], axis=1)
    print(report.sort_values("tokens_raw", ascending=False).to_string())

    df.to_pickle(OUTPUT_PATH)
    print(f"Saved {len(df):,} rows to {OUTPUT_PATH}")
    return df # this will return the df and create a pkl / df file that is already setup


if __name__ == "__main__":
    process()