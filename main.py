from rapidfuzz.distance import Levenshtein
from product import Product
from adapters import load_costco, load_superstore
from llm_processing import gemini_processing, openai_processing
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy

FILE_A = "sample_data\GroceryDataset.csv"
FILE_B = "sample_data\grocery_data_apr_2025-selected-columns.csv"
llm_provider = "gemini"
products_a = load_costco(FILE_A)
products_b = load_superstore(FILE_B)
products_dict_a = {}
products_dict_b = {}
unmodified_names_a = {}
unmodified_names_b = {}
names_a = []
names_b = []
ids_a = []
ids_b = []

confirmed_pairs = []
questionable_pairs = []

def get_brands_regex_compiled(products: list[Product]):
    brands_raw = set()
    for product in products:
        brands_raw.add(str(product.brand))
        print(product.brand)


    brands_processed = sorted(set(b.strip().lower() for b in brands_raw if b.strip()), key=len, reverse=True)
    regex = r"\b(" + "|".join(re.escape(b) for b in brands_processed) + r")\b"
    regex_compiled = re.compile(regex, flags=re.IGNORECASE)
    return regex_compiled

def score_calculator(tfidf_score: float, a_txt: str, b_txt: str):
    # tfidf - 0.8 / leven  - 0.2
    tfidf_multi = 0.8
    leven_score = 0.2
    # Accounts for slight typos. 
    leven_multi = Levenshtein.normalized_similarity(a_txt, b_txt)
    final_score = tfidf_score * tfidf_multi + leven_score * leven_multi
    return final_score
    
def add_match(a_inds: list, a_id: list, b_id: list, b_ind: int, tfidf_scores: list):
    max_id = 0
    max_similarity = -1
    b_cur_id = b_id[b_ind]
    b_clean_name = products_dict_b[b_cur_id].name
    b_full_name = unmodified_names_b[b_cur_id]
    for index, a_ind in enumerate(a_inds):
        a_cur_id = a_id[a_ind]
        a_clean_name = products_dict_a[a_cur_id].name
        tfidf_score = tfidf_scores[index]
        final_score = score_calculator(tfidf_score, a_clean_name, b_clean_name)
        if final_score > max_similarity:
            max_similarity = final_score
            max_id = a_cur_id
    a_full_name = unmodified_names_a[max_id]
    if (len(products_dict_a[max_id].name.split()) < 2 or len(b_clean_name.split()) < 2) and max_similarity > 0.8:
        confirmed_pairs.append((max_id, b_cur_id))
    elif max_similarity > 0.7:
        if not (len(products_dict_a[max_id].name.split()) < 2 or len(b_clean_name.split()) < 2):
            questionable_pairs.append((max_id, b_cur_id))
        else:
            questionable_pairs.append((max_id, b_cur_id))

def process_ambiguous_cases(cases: list, products_dict_a: dict, products_dict_b: dict, confirmed_pairs: list, unmodified_names_a: dict, unmodified_names_b: dict):
    prompt = f"Analyze the following {len(cases)} pairs of products, determine if they are a match based on your system instructions.\n"
    for index, case in enumerate(cases):
        prompt += f"Pair {index}:\n"
        a_id, b_id = case
        product_a = products_dict_a[a_id]
        product_b = products_dict_b[b_id]
        a_name = unmodified_names_a[a_id]
        b_name = unmodified_names_b[b_id]
        a_description = product_a.description
        b_description = product_b.description

        prompt += f"Product A Name: {a_name}\n"
        prompt += f"Product B Name: {b_name}\n"
        prompt += f"Product A Description: {a_description}\n"
        prompt += f"Product B Description: {b_description}\n"
    if llm_provider == "gemini":
        boolean_list = gemini_processing(prompt)
    elif llm_provider == "openai":
        boolean_list = openai_processing(prompt)
    else:
        print("no LLM supplied")
        return
    for result in boolean_list.results:
        if result.is_match and (result.index < len(cases) and 0 <= result.index):
            confirmed_pairs.append(cases[result.index])

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
letter_regex_compiled = re.compile(r"[^a-zA-Z\s]", flags=re.IGNORECASE)
space_regex_compiled = re.compile(r"\s+", flags=re.IGNORECASE)
fluff_word_regex_compiled = re.compile(r"\b(" + "|".join(fluff_word_list) + r")\b", flags=re.IGNORECASE)

for product in products_a:
    unmodified_names_a[product.product_id] = product.name
    product.name = re.sub(fluff_word_regex_compiled, "", product.name)
    product.name = re.sub(letter_regex_compiled, "", product.name)
    product.name = re.sub(space_regex_compiled, " ", product.name).strip()
    names_a.append(product.name)
    ids_a.append(product.product_id)
    products_dict_a[product.product_id] = product
    
for product in products_b:
    unmodified_names_b[product.product_id] = product.name
    product.name = re.sub(fluff_word_regex_compiled, "", product.name)
    product.name = re.sub(letter_regex_compiled, "", product.name)
    product.name = re.sub(space_regex_compiled, " ", product.name).strip()
    names_b.append(product.name)
    ids_b.append(product.product_id)
    products_dict_b[product.product_id] = product

vectorizer = TfidfVectorizer()
vectorizer.fit(names_a + names_b)

vectorized_a = vectorizer.transform(names_a)
vectorized_b = vectorizer.transform(names_b)
for i in range(0, vectorized_b.shape[0], 500):
    chunk = vectorized_b[i : i + 500]
    sim_matrix = cosine_similarity(chunk, vectorized_a)
    top3_matches = numpy.argpartition(sim_matrix, -3, axis=1)[:, -3:]
    for j in range(sim_matrix.shape[0]):
        index = i + j
        item_b_id = ids_b[index]
        a_ind_1, a_ind_2, a_ind_3 = top3_matches[j]
        score_1 = sim_matrix[j, a_ind_1]
        score_2 = sim_matrix[j, a_ind_2]
        score_3 = sim_matrix[j, a_ind_3]
        tfidf_scores = [score_1, score_2, score_3]
        add_match(top3_matches[j], ids_a, ids_b, index, tfidf_scores)

for i in range(0, len(questionable_pairs), 10):
    print(f"Processing LLM batch {i} to {i + 10}")
    process_ambiguous_cases(questionable_pairs[i:i+10], products_dict_a, products_dict_b, confirmed_pairs, unmodified_names_a, unmodified_names_b)

for index, case in enumerate(confirmed_pairs):
    a, b = case
    print(f"{index}. {unmodified_names_a[a]} matched with {unmodified_names_b[b]}")