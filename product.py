from dataclasses import dataclass

@dataclass
class Product:
    product_id: str
    name: str
    description: str = ""
    brand: str = ""
    categories: str = ""
    size: str = ""
    price: str = ""