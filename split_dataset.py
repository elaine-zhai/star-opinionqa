import json
import random
from collections import defaultdict

# Load dataset
with open("modular_pluralism/input/steerable_test_opinionqa.json", "r") as f:
    data = json.load(f)

# Group ALL responses by question
question_responses = defaultdict(list)
for entry in data:
    question_responses[entry["question"]].append(entry)

# Convert to list of questions with their responses
questions = list(question_responses.items())
random.shuffle(questions)  # Shuffle the questions

# Split QUESTIONS (not responses) into 85:5:10
n = len(questions)
train_end = int(0.85 * n)
val_end = train_end + int(0.05 * n)

# Assign whole questions to splits
train_questions = questions[:train_end]
val_questions = questions[train_end:val_end]
test_questions = questions[val_end:]

# Flatten the splits
train = [resp for q, responses in train_questions for resp in responses]
val = [resp for q, responses in val_questions for resp in responses]
test = [resp for q, responses in test_questions for resp in responses]

def save_json(data, filename):
    with open(filename, "w") as f:
        json.dump(data, f, indent=2)

save_json(train, "modular_pluralism/input/opinionqa_train.json")
save_json(val, "modular_pluralism/input/opinionqa_val.json")
save_json(test, "modular_pluralism/input/opinionqa_test.json")

print(f"Split complete: {len(train)} train, {len(val)} val, {len(test)} test")
