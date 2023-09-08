import torch
import time
from transformers import GPT2Tokenizer, GPT2Model

model = GPT2Model.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')


example_input = tokenizer.encode("Once upon a time", return_tensors="pt")
model = torch.jit.freeze(torch.jit.trace(model, (example_input,)))

with torch.no_grad():
    start = time.time()
    output = model.generate(example_input, max_length=50, num_return_sequences=1)
    finish = time.time()
    print(finish - start)