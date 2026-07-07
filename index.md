# Product Matching API

A REST API for large-scale product entity resolution: given two product catalogs, it identifies which rows represent the same real-world item — even across inconsistent naming, typos, and formatting — using a hybrid statistical scoring model with an LLM fallback for genuinely ambiguous cases.

## The problem

Matching product records across catalogs is deceptively hard at scale. The same item can appear as "Premium Organic Whole Milk 1 Gallon" in one dataset and "Whole Milk Gallon" in another — different naming conventions, marketing language, typos, and unit formatting all get in the way of naive matching. This project resolves **230,000 product records against a 55,000-item catalog**, a scale where manual review isn't an option and simple exact/substring matching breaks down constantly.

## How it works

Product names are cleaned (marketing fluff and unit words stripped via regex) and vectorized with TF-IDF. For each item, the top candidate matches are retrieved using chunked cosine similarity — computed in batches to bound memory, with `numpy.argpartition` used instead of a full sort to keep candidate selection fast at scale. Each candidate is then re-scored with a blended metric: 80% TF-IDF cosine similarity plus 20% Levenshtein similarity, which catches near-duplicates with typos that pure TF-IDF (bag-of-words, spelling-insensitive) tends to miss.

Rather than a single hard cutoff, matches are routed by confidence: high-confidence pairs are auto-accepted, low scores are discarded, and the ambiguous middle band is batched and sent to an LLM for a final judgment call using each product's name and description — keeping the expensive step reserved for genuinely uncertain cases instead of running it over the whole dataset.

The matching logic was originally a standalone script; it's since been wrapped in a FastAPI service, containerized with Docker, and deployed live. One design decision worth calling out: rather than trying to auto-detect CSV column formats (fragile across real-world data sources), the API requires callers to explicitly map their columns to the fields it expects — an explicit contract instead of a guess.

## Results

Running against a real 230,000-record dataset matched against a 55,000-item catalog, the pipeline produced **8,600+ validated matches** — confirmed through a combination of automated threshold-based acceptance and manual sample verification.

## Stack

Python · FastAPI · pandas · scikit-learn · RapidFuzz · Docker · Render · LLM APIs (Gemini/OpenAI)

[View source on GitHub →](https://github.com/MOARMOARMAN) · [Live demo →](#)
