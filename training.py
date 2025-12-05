import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel , AutoTokenizer
import time
import math


# model = GPT2LMHeadModel.from_pretrained("gpt2")
# model.train()

tokenizer = AutoTokenizer.from_pretrained("gpt2")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def calculate_loss(model, batch):
#    for batch in train_dataloader:
       input_ids = batch['input_ids']
       attention_mask = batch['attention_mask']
       labels = batch['labels']
       outputs = model(
          input_ids = input_ids,
          attention_mask = attention_mask,
          labels = labels
       )
       loss = outputs.loss
       return loss
    # print(f"Batch loss: {loss.item()}")

def train_epoch(model, dataloader, optimizer, epoch_num , device):
    model.train()
    total_loss = 0
    num_batches = len(dataloader)

    print(f"\n{'='*50}")
    print(f"Epoch {epoch_num} - Training")
    print(f"{'='*50}")
    
    start_time = time.time()

    for batch_idx, batch in enumerate(train_dataloader):
          batch = {k:v.to(device) for k,v in batch.items()}
    # for batch in train_dataloader:
    #       batch = {k: v.to(device) for k, v in batch.items()} 
          loss = calculate_loss(model, batch)

          optimizer.zero_grad()
          loss.backward()

          torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
          optimizer.step()
          total_loss += loss.item()


          if (batch_idx + 1) % 5 == 0 or (batch_idx + 1) == num_batches:
            avg_loss_so_far = total_loss / (batch_idx + 1)
            print(f"  Batch {batch_idx + 1}/{num_batches} | "
                  f"Loss: {loss.item():.4f} | "
                  f"Avg Loss: {avg_loss_so_far:.4f}")
    
    epoch_time = time.time() - start_time
    epoch_loss = total_loss / num_batches
    
    print(f"\nEpoch {epoch_num} Summary:")
    print(f"  Average Training Loss: {epoch_loss:.4f}")
    print(f"  Training Time: {epoch_time:.1f} seconds")
    print(f"  Perplexity: {math.exp(epoch_loss):.2f}")
    
    return epoch_loss


def validate_epoch(model, dataloader, device):
     model.eval()
     total_loss = 0
     num_batches = len(dataloader)
     with torch.no_grad():
          for batch_idx, batch in enumerate(dataloader):
                batch = {k:v.to(device) for k,v in batch.items()}
                loss = calculate_loss(model, batch)
                total_loss += loss.item()

     epoch_loss = total_loss / num_batches
     print(f"Validation Loss: {epoch_loss:.4f}")
     print(f"Validation Perplexity: {math.exp(epoch_loss):.2f}")
     return epoch_loss

def generate_text(model, tokenizer, prompt, max_length=50):
     model.eval()
    #  input_ids = tokenizer.encode(prompt)
    #  input_ids = torch.tensor([input_ids]).to(model.device)
     inputs = tokenizer(prompt, return_tensors="pt")
     input_ids = inputs['input_ids'].to(model.device)

     with torch.no_grad():
            output_ids = model.generate(
                 input_ids,
                 max_length = max_length,
                 num_return_sequences = 1,
                 no_repeat_ngram_size = 2,

            )
     generated_text = tokenizer.decode(output_ids[0], skip_special_tokens = True)
     return generated_text;

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

num_epochs = 3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
best_val_loss = float("inf") 


for epoch in range(1, num_epochs +1):
        train_loss = train_epoch(model, train_dataloader, optimizer, epoch, device)
        val_loss = validate_epoch(model, val_dataloader, device)

        if val_loss < best_val_loss:
           best_val_loss = val_loss
           torch.save({
               'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                }, "best_model_checkpoint.pt")
        print(f"  ✓ Saved best model checkpoint (loss: {val_loss:.4f})")

        scheduler.step()

        print(f"\nSample generation after epoch {epoch + 1}:")
        prompt = "Instruction: Explain quantum physics in simple terms\nResponse:"
        generated = generate_text(model, tokenizer, prompt, max_length=100)
        print(f"  {generated}\n")
    
        print(f"{'='*60}\n")