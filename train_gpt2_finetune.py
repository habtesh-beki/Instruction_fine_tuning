import json
import random
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AdamW
from tqdm import tqdm

# ----------------------------------------------------------
# 1. Load Dataset
# ----------------------------------------------------------

with open("dataset.json", "r") as f:
    data = json.load(f)

random.shuffle(data)

# Train/Val Split
train_data = data[:int(0.9 * len(data))]
val_data = data[int(0.9 * len(data)):]

# ----------------------------------------------------------
# 2. Alpaca Formatting Function
# ----------------------------------------------------------

def format_example(ex):
    instruction = ex["instruction"]
    inp = ex.get("input", "")
    output = ex["output"]

    if inp.strip() == "":
        text = f"### Instruction:\n{instruction}\n\n### Output:\n{output}"
    else:
        text = f"### Instruction:\n{instruction}\n\n### Input:\n{inp}\n\n### Output:\n{output}"

    return text

# ----------------------------------------------------------
# 3. Load Tokenizer & Model
# ----------------------------------------------------------

model_name = "gpt2"   # 124M
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

model = GPT2LMHeadModel.from_pretrained(model_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# ----------------------------------------------------------
# 4. PyTorch Dataset
# ----------------------------------------------------------

class InstructionDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_length=256):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        example = format_example(self.dataset[idx])
        enc = self.tokenizer(
            example,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        input_ids = enc["input_ids"].squeeze()
        attention_mask = enc["attention_mask"].squeeze()

        # labels = input_ids for GPT-2 (causal LM)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids.clone()
        }

train_dataset = InstructionDataset(train_data, tokenizer)
val_dataset = InstructionDataset(val_data, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=2)

# ----------------------------------------------------------
# 5. Training Setup
# ----------------------------------------------------------

optimizer = AdamW(model.parameters(), lr=5e-5)

EPOCHS = 3

# ----------------------------------------------------------
# 6. Training Loop
# ----------------------------------------------------------

print("\nStarting training...\n")

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for batch in tqdm(train_loader):
        batch = {key: val.to(device) for key, val in batch.items()}

        outputs = model(**batch)
        loss = outputs.loss

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()

    avg_train_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1} Train Loss: {avg_train_loss:.4f}")

    # ----------------------
    # Validation
    # ----------------------
    model.eval()
    val_loss = 0

    with torch.no_grad():
        for batch in val_loader:
            batch = {key: val.to(device) for key, val in batch.items()}
            outputs = model(**batch)
            val_loss += outputs.loss.item()

    avg_val_loss = val_loss / len(val_loader)
    print(f"Epoch {epoch+1} Val Loss: {avg_val_loss:.4f}")

    # Save checkpoint
    model.save_pretrained(f"./gpt2-finetuned-epoch{epoch+1}")
    tokenizer.save_pretrained(f"./gpt2-finetuned-epoch{epoch+1}")

print("\nTraining complete!")
