from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import RedirectResponse
import pandas as pd
import io
from product import Product
from main import matching_products

app = FastAPI()

@app.get("/")
def redirect_to_docs():
    return RedirectResponse(url="/docs")

def col_val(row, col):
    return getattr(row, col, "") 

def csv_to_products(
    file_bytes: bytes,
    id_col: str,
    name_col: str,
    description_col: str = "",
    brand_col: str = "",
    categories_col: str = "",
    size_col: str = "",
    price_col: str = "",
) -> list[Product]:
    df = pd.read_csv(io.BytesIO(file_bytes))
    df = df.fillna("")

    required = {id_col, name_col}
    missing = required - set(df.columns)
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing required columns: {missing}")

    products = []
    for _, row in df.iterrows():
        products.append(Product(
            product_id=str(row[id_col]),
            name=str(row[name_col]),
            description=col_val(row, description_col),
            brand=col_val(row, brand_col),
            categories=col_val(row, categories_col),
            size=col_val(row, size_col),
            price=col_val(row, price_col),
        ))
    return products


@app.post("/match-csv")
def match_csv_endpoint(
    file_a: UploadFile = File(...),
    file_b: UploadFile = File(...),
    a_id_col: str = Form(...),
    a_name_col: str = Form(...),
    a_description_col: str = Form(None),
    a_brand_col: str = Form(None),
    b_id_col: str = Form(...),
    b_name_col: str = Form(...),
    b_description_col: str = Form(None),
    b_brand_col: str = Form(None),
):
    products_a = csv_to_products(file_a.file.read(), a_id_col, a_name_col, a_description_col, a_brand_col)
    products_b = csv_to_products(file_b.file.read(), b_id_col, b_name_col, b_description_col, b_brand_col)

    confirmed_pairs, names_a, names_b = matching_products(products_a, products_b)
    return {
        "matches": [{"a": names_a[a], "b": names_b[b]} for a, b in confirmed_pairs],
        "count": len(confirmed_pairs),
    }