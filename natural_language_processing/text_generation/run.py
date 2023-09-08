import torch
import time
from transformers import GPT2Tokenizer, GPT2Model

model = GPT2Model.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

example_input = tokenizer.encode("Once upon a time", return_tensors="pt")
model = torch.jit.freeze(torch.jit.trace(model, (example_input,)))

recorded_time = 0.0
samples = 100
with torch.no_grad():
    for i in range(samples):
        start = time.time()
        output = model.generate(example_input, max_length=50, num_return_sequences=1)
        finish = time.time()
        recorded_time += (finish - start)

print('average latency per sample: (s)', recorded_time/samples)
print('throughput: (fps)', samples/recorded_time)
