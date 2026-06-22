from openai import OpenAI
import yaml
from .batch_result import batch_result
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file).get("gemini")
client = OpenAI(
        api_key=config.get("api_key")
    )   
def openai_processing(prompt: str):
    try:
        completion = client.chat.completions.parse(
            model=config.get("deployment_name"),
            temperature=0.0,
            response_format=batch_result,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a product-matching system. For each pair, you must return a boolean is_match and its index. "
                        "The name has greater weight on matching. The categories and tags are secondary. "
                        "**IMPORTANT** YOU MUST ANALYZE EVERY PAIR AND RETURN A index_result FOR EACH PAIR."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
        )
        boolean_list = completion.choices[0].message.parsed.results
        return boolean_list
    except Exception as e:
        print(f"batch failed because of {e}")
        return
