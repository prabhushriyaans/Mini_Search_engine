import torch
import torch.nn as nn
import torch.optim as optim
from collections import Counter
import json
import os

# -----------------------------
# 0. DEVICE
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

FILE_NAME = "corpus.json"

# -----------------------------
# 1. FILE HANDLING
# -----------------------------
def load_corpus():
    if not os.path.exists(FILE_NAME):
        return []
    with open(FILE_NAME, "r") as f:
        return json.load(f).get("sentences", [])

def save_corpus(corpus):
    with open(FILE_NAME, "w") as f:
        json.dump({"sentences": corpus}, f, indent=4)

# -----------------------------
# 2. LOAD CORPUS
# -----------------------------
corpus = load_corpus()

# -----------------------------
# 3. BUILD WORD DATA
# -----------------------------
def build_data(corpus):
    words = []
    for sentence in corpus:
        words.extend(sentence.lower().split())

    word_counts = Counter(words)
    vocab = {word: i+1 for i, (word, _) in enumerate(word_counts.items())}
    index_to_word = {i: w for w, i in vocab.items()}
    vocab_size = len(vocab) + 1

    sequences = []
    for sentence in corpus:
        token_list = [vocab[word] for word in sentence.lower().split()]
        for i in range(1, len(token_list)):
            sequences.append(token_list[:i+1])

    if len(sequences) == 0:
        return None, None, None, None, None, None

    max_len = max(len(seq) for seq in sequences)

    def pad(seq):
        return [0]*(max_len - len(seq)) + seq

    sequences = torch.tensor([pad(seq) for seq in sequences]).to(device)

    X = sequences[:, :-1]
    y = sequences[:, -1]

    return vocab, index_to_word, vocab_size, X, y, max_len

# -----------------------------
# 4. CHARACTER VOCAB
# -----------------------------
def build_char_vocab(corpus):
    chars = set()
    for sentence in corpus:
        for word in sentence.lower().split():
            chars.update(list(word))

    char2idx = {c: i+1 for i, c in enumerate(chars)}
    idx2char = {i: c for c, i in char2idx.items()}

    return char2idx, idx2char, len(char2idx)+1

def word_to_char_tensor(word, char2idx, max_word_len=10):
    char_ids = [char2idx.get(c, 0) for c in word[:max_word_len]]
    char_ids = [0]*(max_word_len - len(char_ids)) + char_ids
    return torch.tensor(char_ids).to(device)

# -----------------------------
# 5. MODEL
# -----------------------------
class HybridModel(nn.Module):
    def __init__(self, vocab_size, char_vocab_size):
        super().__init__()

        # WORD
        self.embedding = nn.Embedding(vocab_size, 64)
        self.lstm = nn.LSTM(64, 128, batch_first=True)

        # CHAR
        self.char_embedding = nn.Embedding(char_vocab_size, 32)
        self.char_lstm = nn.LSTM(32, 64, batch_first=True)

        # TRANSFORMER
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=64,
            nhead=4,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # FINAL
        self.fc = nn.Linear(128 + 64 + 64, vocab_size)

    def forward(self, x, char_x):
        # WORD
        emb = self.embedding(x)
        lstm_out, _ = self.lstm(emb)
        lstm_out = lstm_out[:, -1, :]

        trans_out = self.transformer(emb)
        trans_out = trans_out[:, -1, :]

        # CHAR
        char_emb = self.char_embedding(char_x)
        char_out, _ = self.char_lstm(char_emb)
        char_out = char_out[:, -1, :]

        # COMBINE
        combined = torch.cat((lstm_out, trans_out, char_out), dim=1)
        output = self.fc(combined)

        return output

# -----------------------------
# 6. INITIAL BUILD
# -----------------------------
vocab, index_to_word, vocab_size, X, y, max_len = build_data(corpus)

if vocab is None:
    model = None
    char2idx, idx2char, char_vocab_size = {}, {}, 0
else:
    char2idx, idx2char, char_vocab_size = build_char_vocab(corpus)
    model = HybridModel(vocab_size, char_vocab_size).to(device)

# -----------------------------
# 7. TRAIN
# -----------------------------
def train_model(epochs=200):
    global model

    if model is None:
        print("No data to train.")
        return

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        # build char input (last word of each sequence)
        char_inputs = []
        for seq in X:
            word_idx = seq[-1].item()
            word = index_to_word.get(word_idx, "")
            char_inputs.append(word_to_char_tensor(word, char2idx))

        char_inputs = torch.stack(char_inputs)

        output = model(X, char_inputs)
        loss = criterion(output, y)

        loss.backward()
        optimizer.step()

        if epoch % 20 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")

# -----------------------------
# 8. PREDICT
# -----------------------------
def predict_top_k(text, k=3):
    if model is None:
        return ["No model trained"]

    model.eval()

    tokens = text.lower().split()
    last_word = tokens[-1]

    token_ids = [vocab.get(word, 0) for word in tokens]
    token_ids = [0]*(max_len - len(token_ids)) + token_ids

    input_tensor = torch.tensor([token_ids[:-1]]).to(device)
    char_tensor = word_to_char_tensor(last_word, char2idx).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor, char_tensor)
        topk = torch.topk(output, k)
        indices = topk.indices[0].tolist()

    return [index_to_word.get(i, "") for i in indices]

# -----------------------------
# 9. SAVE / LOAD
# -----------------------------
def save_model():
    if model:
        torch.save(model.state_dict(), "hybrid_model.pth")
        print("Saved!")

def load_model():
    global model
    if model:
        try:
            model.load_state_dict(torch.load("hybrid_model.pth", map_location=device))
            model.to(device)
            print("Loaded!")
        except:
            print("No saved model found")

# -----------------------------
# 10. RETRAIN
# -----------------------------
def retrain():
    global vocab, index_to_word, vocab_size, X, y, max_len
    global char2idx, idx2char, char_vocab_size, model

    data = build_data(corpus)
    if data[0] is None:
        print("Not enough data")
        return

    vocab, index_to_word, vocab_size, X, y, max_len = data
    char2idx, idx2char, char_vocab_size = build_char_vocab(corpus)

    model = HybridModel(vocab_size, char_vocab_size).to(device)

    print("Training...")
    train_model(300)
    print("Done!")

# -----------------------------
# 11. MENU
# -----------------------------
def menu():
    print("\n==== HYBRID SEARCH ENGINE ====")
    print("1. Predict")
    print("2. Add sentence")
    print("3. Retrain")
    print("4. Save model")
    print("5. Load model")
    print("6. Exit")

# -----------------------------
# 12. RUN LOOP
# -----------------------------
while True:
    menu()
    choice = input("Enter choice: ")

    if choice == "1":
        text = input("Enter text: ")
        k = int(input("Enter k: "))
        print("Predictions:", predict_top_k(text, k))

    elif choice == "2":
        s = input("Enter sentence: ").lower()
        if s not in corpus:
            corpus.append(s)
            save_corpus(corpus)
            print("Saved! Retrain needed.")
        else:
            print("Already exists")

    elif choice == "3":
        retrain()

    elif choice == "4":
        save_model()

    elif choice == "5":
        load_model()

    elif choice == "6":
        break

    else:
        print("Invalid choice")