import json
import random

SIZE =5000
# Olası değerler
labels = ["OTHER", "SEXIST", "RACIST", "INSULT", "PROFANITY"]

# Veri oluşturma
data = [
    {"target": random.choice(labels), "predict": random.choice(labels)}
    for _ in range(SIZE)
]

file_path = "example_output.json"

with open(file_path, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=4, ensure_ascii=False)

print(f"{file_path} dosyası oluşturuldu.")
