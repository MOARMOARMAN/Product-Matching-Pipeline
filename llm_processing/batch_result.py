from typing import List
from pydantic import BaseModel
class index_result(BaseModel):
    is_match: bool
    index: int
class batch_result(BaseModel):
    results: List[index_result]