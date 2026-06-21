from openai import OpenAI
import yaml
import pandas as pd
import json
import re
import numpy
from pydantic import BaseModel
from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz.distance import Levenshtein
import sys
import io

BATCH_SIZE = 10000
FILE_A = "grocery_store_a_items_final.csv"
FILE_B = "grocery_store_b_items_final.csv"
with open("openai_creds.yaml", "r") as file:
    openai_creds = yaml.safe_load(file).get("openai")

client = OpenAI(
    base_url=openai_creds.get("endpoint"),
    api_key=openai_creds.get("api_key")
)

# Enforce the output for LLM
class index_result(BaseModel):
    is_match: bool
    index: int
class batch_result(BaseModel):
    results: List[index_result]

def convert_str_to_dict(s: str):
    if not s or pd.isna(s):
        return {}
    try: 
        return json.loads(s)
    except json.JSONDecodeError:
        print(f"This entry is corrupted {s}")
        return {}

def get_category_paths(listing: dict):
    info = listing.get("info", {})
    cats = [info.get("category_0", ""), info.get("category_1", ""), info.get("category_2", ""), info.get("category_3", "")]
    return ">".join([x for x in cats if x])

def get_brands_regex_compiled(file_name: str):
    brands_raw = set()
    for batch in pd.read_csv(file_name, usecols=["brand_raw"], chunksize=BATCH_SIZE):
        brands_raw.update(str(b) for b in batch["brand_raw"])

    brands_processed = sorted(set(b.strip().lower() for b in brands_raw if b.strip()), key=len, reverse=True)
    regex = r"\b(" + "|".join(re.escape(b) for b in brands_processed) + r")\b"
    regex_compiled = re.compile(regex)
    return regex_compiled

def populate_listings_dict(listings: dict, file_name: str, relevant_categories: list, brand_regex: str, text_lst: list, id_lst: list):
    for batch in pd.read_csv(file_name, chunksize=BATCH_SIZE, usecols=relevant_columns):
        batch = batch[batch["item_id"].notna()]
        batch["item_info"] = batch["item_info"].apply(
            lambda x: convert_str_to_dict(x)   
        )
        batch = batch[batch["item_info"].str["category_0"].isin(relevant_categories)]
        batch["sizing_comp"] = batch["sizing_comp"].apply(
            lambda x: convert_str_to_dict(x)   
        )
        batch["user_friendly_size"] = batch["sizing_comp"].str["size_user_friendly"].fillna("").astype(str)
        batch["clean_name"] = (  
            batch["name"]
            .astype(str)
            .str.lower()
            .str.strip()
            .str.replace(brand_regex, "", regex=True)
            .str.replace(letter_regex_compiled, "", regex=True)
            .str.replace(fluff_word_regex_compiled, "", regex=True)
            .str.replace(space_regex_compiled, " ", regex=True)
            .str.strip()
        )

        zipped_data = zip(
            batch["item_id"],
            batch["name"],
            batch["clean_name"],
            batch["tags"],
            batch["user_friendly_size"],
            batch["item_info"]
        )

        for item_id, name, clean_name, tags, user_friendly_size, item_info in zipped_data:
            listings[item_id] = {
                "name":name,
                "clean_name":clean_name,
                "tags":tags,
                "size":user_friendly_size,
                "info":item_info
            }
            text_lst.append(clean_name)
            id_lst.append(item_id)

def add_match(a_ind: int, a_listings: dict, a_id: list, b_listings: dict, b_id: list, b_inds: list, tfidf_scores: list, confirmed_pairs: list, questionable_pairs: list):
    max_id = 0
    max_similarity = -1
    a_cur_id = a_id[a_ind]
    a_clean_name = a_listings[a_cur_id]["clean_name"]
    a_name = a_listings[a_cur_id]["name"]
    a_size = a_listings[a_cur_id]["size"]
    for index, b_ind in enumerate(b_inds):
        b_cur_id = b_id[b_ind]
        b_clean_name = b_listings[b_cur_id]["clean_name"]
        b_size = b_listings[b_cur_id]["size"]
        tfidf_score = tfidf_scores[index]
        final_score = score_calculator(tfidf_score, a_size, b_size, a_clean_name, b_clean_name)
        if final_score > max_similarity:
            max_similarity = final_score
            max_id = b_cur_id
    b_name = b_listings[max_id]["name"]
    if (len(a_clean_name.split()) < 2 or len(b_listings[max_id]["clean_name"].split()) < 2) and max_similarity > 0.9:
        questionable_pairs.append(((a_cur_id, max_id)))
    elif max_similarity > 0.9:
        confirmed_pairs.append((a_cur_id, max_id))
    elif max_similarity > 0.85:
        if not (len(a_clean_name.split()) < 2 or len(b_listings[max_id]["clean_name"].split()) < 2):
            confirmed_pairs.append((a_cur_id, max_id))
        else:
            questionable_pairs.append((a_cur_id, max_id))

def score_calculator(tfidf_score: float, size_txt_a: str, size_txt_b: str, a_txt: str, b_txt: str):
    # tfidf - 0.8 / size - 0.1 / leven  - 0.1
    tfidf_multi = 0.8
    size_score = 0.1
    leven_score = 0.1

    size_txt_a = size_txt_a.lower().replace(" ", "")
    size_txt_b = size_txt_b.lower().replace(" ", "")
    # size is probably not a big impact on whether products are different
    # If a product is available in a smaller quantity people will still purchase it if they want it.
    if not size_txt_b or not size_txt_a:
        size_multi = 0.8 # if size is unknown don't penalize too much.
    elif size_txt_b == size_txt_a:
        size_multi = 1
    else:
        size_multi = 0.6 # even if size is different it shouldn't be too much of an issue.
    # Accounts for slight typos. 
    leven_multi = Levenshtein.normalized_similarity(a_txt, b_txt)
    final_score = tfidf_score * tfidf_multi + size_score * size_multi + leven_score * leven_multi
    return final_score

def process_ambiguous_cases(cases: list, a_listings: dict, b_listings: dict, confirmed_pairs: list):
    prompt = f"Analyze the following {len(cases)} pairs of products, determine if they are a match based on your system instructions.\n"
    for index, case in enumerate(cases):
        prompt += f"Pair {index}:\n"
        a_id, b_id = case
        listing_a = a_listings[a_id]
        listing_b = b_listings[b_id]
        a_name = listing_a["name"]
        b_name = listing_b["name"]
        a_tags = listing_a.get("tags", "N/A")
        b_tags = listing_b.get("tags", "N/A")
        a_cats = get_category_paths(listing_a)
        b_cats = get_category_paths(listing_b)
        
        prompt += f"Product A Name: {a_name}\n"
        prompt += f"Product B Name: {b_name}\n"
        prompt += f"Product A Tags: {a_tags}\n"
        prompt += f"Product B Tags: {b_tags}\n"
        prompt += f"Product A Categories: {a_cats}\n"
        prompt += f"Product B Categories: {b_cats}\n"

    try:
        completion = client.chat.completions.parse(
            model=openai_creds.get("deployment_name"),
            temperature=0.0,
            response_format=batch_result,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a product-matching system. For each pair, you must return a boolean is_match and its index. "
                        "The name has greater weight on matching. The categories and tags are secondary. "
                        "**IMPORTANT** YOU MUST ANALYZE EVERY PAIR AND RETURN A index_result FOR EACH PAIR."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
        )
        boolean_list = completion.choices[0].message.parsed.results
        for result in boolean_list:
            if result.is_match and (result.index < len(cases) and 0 <= result.index):
                confirmed_pairs.append(cases[result.index])
    except Exception as e:
        print(f"Batch failed because of {e}")
        return

if __name__ == "__main__":
    relevant_columns = ['item_id', 'name', 'brand_raw', 'item_info', 'tags', 'sizing_comp']
    fluff_word_list = ["original", "classic", "premium", "select", "choice", "prime",
                       "natural", "real", "pure", "new", "improved", "best", "great",
                       "ultimate", "legendary", "with", "for", "up to", "by", "in",
                       "ultra", "fluid ounces", "fluid ounce", "fluid oz", "fl oz", 
                        "milliliters", "milliliter", "kilograms", "kilogram", "packages", 
                        "package", "pieces", "piece", "ounces", "ounce", "grams", "gram", 
                        "liters", "liter", "pounds", "pound", "quarts", "quart", "pints", 
                        "pint", "gallons", "gallon", "ml", "cl", "kg", "mg", "lb", "lbs", 
                        "oz", "qts", "qt", "pts", "pt", "gal", "ct", "pk", "pcs", "g", "l",
                        "super", "naturally", "box", "a", "pack", "packs", "variety", "count"]
    fluff_word_list.sort(key=len, reverse=True)
    relevant_categories_a = ['Food']
    relevant_categories_b = ['Grocery', 'Frozen', 'Seafood', 'Cheese', 'Meat', 'Produce & Floral', 'Wine, Beer & Spirits', 'Prepared Foods', 'Bakery', 'Dairy']
    brand_regex_compiled_a = get_brands_regex_compiled(FILE_A)
    brand_regex_compiled_b = get_brands_regex_compiled(FILE_B)
    letter_regex_compiled = re.compile(r"[^a-zA-Z\s]")
    space_regex_compiled = re.compile(r"\s+")
    fluff_word_regex_compiled = re.compile(r"\b(" + "|".join(fluff_word_list) + r")\b")

    a_listings = {}
    b_listings = {}
    a_id = []
    a_text = []
    b_id = []
    b_text = []
    
    confirmed_pairs = []
    questionable_pairs = []

    populate_listings_dict(a_listings, FILE_A, relevant_categories_a, brand_regex_compiled_a, a_text, a_id)
    populate_listings_dict(b_listings, FILE_B, relevant_categories_b, brand_regex_compiled_b, b_text, b_id)
    # Calculate similarity score
    # 1. TF IDF (Term Frequency - Inverse Document Frequency)
    # 2. Size
    # 3. Levenshtein distance (number of character edits to reach target)
    vectorizer = TfidfVectorizer()
    vectorizer.fit(a_text + b_text)

    # Maps each word in each product name to how "important" it is accoriding to TF-IDF
    vectorized_a = vectorizer.transform(a_text)
    vectorized_b = vectorizer.transform(b_text)
    for row in range(0, vectorized_a.shape[0], 500):
        chunk = vectorized_a[row : row + 500]
        similarity_matrix = cosine_similarity(chunk, vectorized_b)
        top3_matches = numpy.argpartition(similarity_matrix, -3, axis=1)[:, -3:]
        
        for i in range(similarity_matrix.shape[0]):
            index = row + i
            item_a_id = a_id[index]
            b_ind_1, b_ind_2, b_ind_3 = top3_matches[i]

            match_score_1 = similarity_matrix[i, b_ind_1]
            match_score_2 = similarity_matrix[i, b_ind_2]
            match_score_3 = similarity_matrix[i, b_ind_3]
            tfidf_scores = [match_score_1, match_score_2, match_score_3]
            add_match(index, a_listings, a_id, b_listings, b_id, top3_matches[i], tfidf_scores, confirmed_pairs, questionable_pairs)
    
    print(f"Number of Confirmed Pairs Pre-LLM: {len(confirmed_pairs)}")
    # Reduced stepping to 20 because LLM started hallucinating
    for i in range(0, len(questionable_pairs), 25):
        print(f"Processing LLM batch {i} to {i + 25}")
        process_ambiguous_cases(questionable_pairs[i:i+25], a_listings, b_listings, confirmed_pairs)
    print(f"Number of Confirmed Pairs Post-LLM: {len(confirmed_pairs)}")

    df_output = pd.DataFrame(confirmed_pairs, columns=["item_id_A", "item_id_B"])
    df_output.to_csv("matching_products.csv", index=False, encoding="utf-8")