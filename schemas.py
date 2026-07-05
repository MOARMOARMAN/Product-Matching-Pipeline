from pydantic import BaseModel
from typing import List

class ProductIn(BaseModel):
    product_id: str
    name: str
    description: str = ""
    brand: str = ""
    categories: str = ""
    size: str = ""
    price: str = ""

class MatchRequest(BaseModel):
    products_a: List[ProductIn]
    products_b: List[ProductIn]