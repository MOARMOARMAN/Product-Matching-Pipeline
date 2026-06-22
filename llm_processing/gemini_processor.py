import yaml
from google import genai
from google.genai import types
from .batch_result import batch_result

with open("config.yaml", "r") as file:
    config = yaml.safe_load(file).get("gemini")
client = genai.Client(api_key=config.get("api_key")) 
def gemini_processing(prompt: str):
    instructions =  ("You are a product-matching system. For each pair, you must return a boolean is_match and its index. "
                    "The name has greater weight on matching. The categories and tags are secondary. "
                    "**IMPORTANT** YOU MUST ANALYZE EVERY PAIR AND RETURN A index_result FOR EACH PAIR.")
    try:
        completion = client.models.generate_content(
            model=config.get("deployment_name"),
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=instructions,
                response_mime_type="application/json",
                response_schema=batch_result,
                temperature=0
            )
        )
        boolean_list = completion.parsed
        print(boolean_list)
        return boolean_list
    except Exception as e:
        print(f"Batch failed because of {e}")
        return