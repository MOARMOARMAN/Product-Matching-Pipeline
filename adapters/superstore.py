import pandas as pd
from product import Product

def load_superstore(file_name: str):
    products = []
    for batch in pd.read_csv(file_name, chunksize=5000):
        for row in batch.itertuples(index=False):
            products.append(Product(
                product_id=getattr(row, "productId", ""),
                name=getattr(row, "title", ""),
                description=getattr(row, "description", ""),
                brand=getattr(row, "brand", ""),
                categories=getattr(row, "Sub Category", ""),
                size=getattr(row, "packageSizing", ""),
                price=getattr(row, "Price", "")
            ))  
    
    return products