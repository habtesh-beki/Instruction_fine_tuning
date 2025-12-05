from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Load pretrained GPT-2 (124M)
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

model.eval()

prompt = "I am a developer"
inputs = tokenizer(prompt, return_tensors="pt")

with torch.no_grad():
    output = model.generate(
        **inputs,
        max_length=40,
        do_sample=True,
        temperature=1.0,
        top_p=0.9
    )

print(tokenizer.decode(output[0]))
