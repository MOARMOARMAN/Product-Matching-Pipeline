from adapters import load_costco, load_superstore

FILE_A = "sample_data\GroceryDataset.csv"
FILE_B = "sample_data\grocery_data_apr_2025-selected-columns.csv"
print(load_costco(FILE_A))
print(load_superstore(FILE_B))


