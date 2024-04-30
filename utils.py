import re

functional_token_mapping = {
  "<nexa_0>": "physics_gpt",
  "<nexa_1>": "chemistry_gpt",
  "<nexa_2>": "biology_gpt",
  "<nexa_3>": "computer_science_gpt",
  "<nexa_4>": "math_gpt",
  "<nexa_5>": "electrical_engineering_gpt",
  "<nexa_6>": "history_gpt",
  "<nexa_7>": "philosophy_gpt",
  "<nexa_8>": "law_gpt",
  "<nexa_9>": "politics_gpt",
  "<nexa_10>": "culture_gpt",
  "<nexa_11>": "economics_gpt",
  "<nexa_12>": "geography_gpt",
  "<nexa_13>": "psychology_gpt",
  "<nexa_14>": "business_gpt",
  "<nexa_15>": "health_gpt",
  "<nexa_16>": "general_gpt"
}

def extract_content(data):
    pattern = r"<nexa_([0-9]{1,2})>\s*\('([^']*)'\)<nexa_end>"
    matches = re.findall(pattern, data)
    functional_token = f"<nexa_{matches[0][0]}>"
    return functional_token, matches[0][1]

if __name__ == "__main__":
    example = "<nexa_4>('Determine the derivative of the function f(x) = x^3 at the point where x equals 2, and interpret the result within the context of rate of change and tangent slope.')<nexa_end>"
    functional_token, format_argument = extract_content(example)
    print(functional_token)
    print(format_argument)