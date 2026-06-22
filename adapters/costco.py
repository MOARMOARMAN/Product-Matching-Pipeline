import pandas as pd
from product import Product

def load_costco(file_name: str):
    products = []
    for batch in pd.read_csv(file_name, chunksize=5000):
        for row in batch.itertuples(index=False):
            products.append(Product(
                product_id=getattr(row, "Id", ""),
                name=getattr(row, "Title", ""),
                description=getattr(row, "Product Description", ""),
                brand=getattr(row, "brand", ""),
                categories=getattr(row, "Sub Category", ""),
                size=getattr(row, "Feature", ""),
                price=getattr(row, "Price", "")
            ))  
    
    return products

        