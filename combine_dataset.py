## Combines train and val datasets
import json

with open("modular_pluralism/input/opinionqa_train.json") as f1, open("modular_pluralism/input/opinionqa_val.json") as f2:
    train = json.load(f1)
    val = json.load(f2)

merged = train + val

with open("modular_pluralism/input/opinionqa_trainval.json", "w") as fout:
    json.dump(merged, fout, indent=2)

print(f"✅ Merged {len(train)} train and {len(val)} val examples.")

