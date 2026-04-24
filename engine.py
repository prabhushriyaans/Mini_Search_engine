import json
from collections import defaultdict
import random
import os

# -----------------------------
# 1. FILE HANDLING
# -----------------------------
FILE_NAME = "corpus.json"

def load_corpus():
    if not os.path.exists(FILE_NAME):
        return []
    with open(FILE_NAME, "r") as f:
        data = json.load(f)
        return data.get("sentences", [])

def save_corpus(corpus):
    with open(FILE_NAME, "w") as f:
        json.dump({"sentences": corpus}, f, indent=4)

# -----------------------------
# 2. N-GRAM MODEL
# -----------------------------
class NGramModel:
    def __init__(self, n=2):
        self.n = n
        self.model = defaultdict(list)

    def train(self, corpus):
        for sentence in corpus:
            tokens = sentence.lower().split()
            for i in range(len(tokens) - self.n + 1):
                key = tuple(tokens[i:self.n+i-1])
                next_word = tokens[i+self.n-1]
                self.model[key].append(next_word)

    def predict(self, text):
        tokens = text.lower().split()
        key = tuple(tokens[-(self.n-1):])
        if key in self.model:
            return random.choice(self.model[key])
        return None

    def generate(self, text, length=5):
        result = text.split()
        for _ in range(length):
            next_word = self.predict(" ".join(result))
            if not next_word:
                break
            result.append(next_word)
        return " ".join(result)

# -----------------------------
# 3. LOAD DATA
# -----------------------------
corpus = load_corpus()

ngram = NGramModel(n=2)
ngram.train(corpus)

# -----------------------------
# 4. MENU SYSTEM
# -----------------------------
def menu():
    print("\n====== MINI SEARCH ENGINE (N-GRAM + JSON) ======")
    print("1. Generate text")
    print("2. Predict next word")
    print("3. Add new sentence (learn + save)")
    print("4. Show model keys")
    print("5. Exit")

while True:
    menu()
    choice = input("Enter your choice: ")

    if choice == "1":
        text = input("Enter starting text: ")
        length = int(input("How many words to generate: "))
        result = ngram.generate(text, length)
        print("Generated:", result)

    elif choice == "2":
        text = input("Enter text: ")
        next_word = ngram.predict(text)
        print("Next word:", next_word if next_word else "No prediction")

    elif choice == "3":
        new_sentence = input("Enter new sentence: ")
        
        # Update memory
        corpus.append(new_sentence)
        ngram.train([new_sentence])
        
        # Save to JSON
        save_corpus(corpus)
        
        print("Sentence learned and saved! 💾")

    elif choice == "4":
        print("\nModel learned patterns (sample):")
        count = 0
        for k, v in ngram.model.items():
            print(f"{k} -> {v}")
            count += 1
            if count == 10:
                break

    elif choice == "5":
        print("Exiting... 👋")
        break

    else:
        print("Invalid choice, try again!")