import os
import json
import time
import random
from tqdm import tqdm
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed

client = OpenAI()

INPUT_FILE = "modular_pluralism/input/opinionqa_trainval.json"  # merged train+val
COT_OUTPUT = "modular_pluralism/debug/star_opinionqa_rawcot.jsonl"
FINAL_OUTPUT = "modular_pluralism/output/star_opinionqa_synthetic_trainval.jsonl"

os.makedirs(os.path.dirname(COT_OUTPUT), exist_ok=True)
os.makedirs(os.path.dirname(FINAL_OUTPUT), exist_ok=True)

def build_star_prompt(question, options, party, education):
    return f"""You are simulating the perspective of a person with the following demographics:
- Political affiliation: {party}
- Education level: {education}

Answer the following survey question from this perspective. Provide a paragraph of reasoning, then select the best option.

{question}
Options: {', '.join(options)}

Respond in this format:
Chain of Thought: ...
Final Answer: (A/B/C/D)
"""

def build_justification_prompt(question, options, correct_text):
    return f"""The correct answer to the following question is: {correct_text}

{question}
Options: {', '.join(options)}

Please provide a paragraph explaining why this is the correct answer.
"""

def parse_final_answer(reply):
    for line in reversed(reply.strip().split("\n")):
        if "Final Answer" in line:
            ans = line.split("Final Answer:")[-1].strip().replace("(", "").replace(")", "")
            if ans in {"A", "B", "C", "D"}:
                return ans
    return None

def process_item(item):
    item_id = item["id"]
    question = item["question"].strip()
    options = item["options"]
    attr = item.get("attribute", "")
    gold_distribution = item["gold_distribution"]

    correct_index = gold_distribution.index(max(gold_distribution))
    correct_letter = chr(ord("A") + correct_index)
    correct_valence = options[correct_index]

    # Parse demographic attributes
    party, education = "Unknown", "Unknown"
    if attr.startswith("POLPARTY_") or attr.startswith("POLIDEOLOGY_"):
        party = attr.split("_", 1)[-1]
    elif attr.startswith("EDUCATION_"):
        education = attr.split("_", 1)[-1]

    try:
        prompt = build_star_prompt(question, options, party, education)
        response = client.chat.completions.create(
            model="gpt-4.1-mini-2025-04-14",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=1000
        )
        reply = response.choices[0].message.content.strip()
        predicted = parse_final_answer(reply)

        # Save raw COT for debugging
        raw_debug = {
            "id": item_id,
            "prompt": prompt,
            "response": reply,
            "predicted": predicted,
            "label": correct_letter,
            "valence": correct_valence
        }

        with open(COT_OUTPUT, "a") as logf:
            logf.write(json.dumps(raw_debug) + "\n")

        # If model predicted the correct answer, keep it
        if predicted == correct_letter:
            return {
                "id": item_id,
                "cot": reply,
                "correct_valence_option": predicted,
                "valence": options[ord(predicted) - ord("A")],
                "label": correct_letter
            }

        # If model failed, fallback to justification
        j_prompt = build_justification_prompt(question, options, correct_valence)
        j_response = client.chat.completions.create(
            model="gpt-4.1-mini-2025-04-14",
            messages=[{"role": "user", "content": j_prompt}],
            temperature=0.7,
            max_tokens=600
        )
        justification = j_response.choices[0].message.content.strip()

        return {
            "id": item_id,
            "cot": justification,
            "correct_valence_option": correct_letter,
            "valence": correct_valence,
            "label": correct_letter
        }

    except Exception as e:
        print(f"[ERROR] ID {item_id}: {e}")
        return None

def main():
    with open(INPUT_FILE) as f:
        data = json.load(f)

    random.seed(42)
    with open(FINAL_OUTPUT, "w") as fout:
        with ThreadPoolExecutor(max_workers=64) as executor:
            futures = {executor.submit(process_item, item): item for item in data}
            for future in tqdm(as_completed(futures), total=len(futures)):
                result = future.result()
                if result:
                    fout.write(json.dumps(result) + "\n")

    print(f"\n✅ Finished writing: {FINAL_OUTPUT}")

if __name__ == "__main__":
    main()

