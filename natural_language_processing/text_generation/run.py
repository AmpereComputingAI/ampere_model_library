import torch
import time
from transformers import GPT2Tokenizer, GPT2Model

model = GPT2Model.from_pretrained('gpt2', torchscript=True)
model.eval()
model = torch.jit.freeze(torch.jit.script(model))


tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

text = "Replace me by any text you'd like."
encoded_input = tokenizer(text, return_tensors='pt')

traced_model = torch.jit.trace(model, (encoded_input,))
frozen_model = torch.jit.freeze(traced_model)

with torch.no_grad():
    start = time.time()
    output = model.generate(encoded_input, max_length=50, num_return_sequences=1)
    finish = time.time()
    print(finish - start)
