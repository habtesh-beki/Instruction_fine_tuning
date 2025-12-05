import json
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from transformers import AutoTokenizer , GPT2LMHeadModel

tokenizer = AutoTokenizer.from_pretrained("gpt2")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = GPT2LMHeadModel.from_pretrained("gpt2")
model = model.to(device)
# model.train()

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token 

class InstructionDataset(torch.utils.data.Dataset):
    def __init__(self,path, tokenizer, max_length = 512):
        self.samples = []
        self.tokenizer = tokenizer
        self.max_length = max_length

        with open(path, 'r') as f:
            for line in f:
                item = json.loads(line)
                text = item['text']
                self.samples.append(text)
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        text = self.samples[idx]
        encoding = self.tokenizer(
            text,
            truncation = True,
            max_length = self.max_length,
            padding = 'max_length',
            return_tensors = 'pt'
        )
        return {'input_ids': encoding["input_ids"].squeeze(),
               'attention_mask': encoding["attention_mask"].squeeze()}

def collate_fn(batch):
    input_ids = [item['input_ids'] for item in batch]
    attention_mask = [item['attention_mask'] for item in batch]

    padded = tokenizer.pad(
        {'input_ids': input_ids, 'attention_mask': attention_mask},
        padding = True,
        return_tensors = 'pt'
    )

    lables = padded['input_ids'].clone()

    lables[padded['attention_mask'] == 0] = -100
    padded['labels'] = lables

    return padded

train_dataset = InstructionDataset("train_fixed.jsonl", tokenizer)
train_subset = Subset(train_dataset, indices=range(10000)) 
val_dataset = InstructionDataset("val.jsonl", tokenizer)
test_dataset = InstructionDataset("test.jsonl", tokenizer)


train_dataloader = DataLoader(
    # train_dataset,
    train_subset,
    batch_size = 8,
    shuffle = True,
    collate_fn = collate_fn
   )

val_dataloader = DataLoader(
    val_dataset,
    batch_size = 8,
    shuffle = False,
    collate_fn = collate_fn
   )


