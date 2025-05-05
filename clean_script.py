import json

INPUT_FILE = "modular_pluralism/output/star_opinionqa_synthetic_trainval.jsonl"
CLEANED_JSONL = "modular_pluralism/output/star_cleaned.jsonl"
FINAL_JSON = "modular_pluralism/output/trainval.json"

valid_options = {"A", "B"}
cleaned = []

with open(INPUT_FILE, "r") as f_in, open(CLEANED_JSONL, "w") as f_out:
    for line in f_in:
        item = json.loads(line)
        if item.get("correct_valence_option") in valid_options:
            cleaned.append(item)
            f_out.write(json.dumps(item) + "\n")
        else:
            print(f"⚠️ Skipping ID {item.get('id')} with invalid option: {item.get('correct_valence_option')}")

with open(FINAL_JSON, "w") as f_json:
    json.dump(cleaned, f_json, indent=2)

print(f"✅ Cleaned {len(cleaned)} valid entries → {FINAL_JSON}")

