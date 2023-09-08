import os
import torch
import time
from transformers import GPT2Tokenizer, GPT2Model

try:
    omp_num_threads = int(os.environ["AIO_NUM_THREADS"])
    torch.set_num_threads(omp_num_threads)
except KeyError:
    omp_num_threads = None
    print('please set AIO_NUM_THREADS')
    quit()

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2Model.from_pretrained('gpt2', torchscript=True)


text = "Hi, how are you?"
encoded_input = tokenizer(text, return_tensors='pt')
input_dict = {key: value for key, value in encoded_input.items()}

traced_model = torch.jit.trace(model, (input_dict['input_ids'],))
#traced_model = torch.jit.trace(model, (encoded_input,))
frozen_model = torch.jit.freeze(traced_model)

#output = frozen_model(**encoded_input)

# RUN WARMUP
with torch.no_grad():
    for i in range(3):
        output = frozen_model(input_dict['input_ids'])

recorded_time = 0.0
samples = 1000
with torch.no_grad():
    for i in range(samples):
        start = time.time()
        output = frozen_model(input_dict['input_ids'])
        finish = time.time()
        recorded_time += (finish - start)

print('average latency per sample: (s)', recorded_time / samples)
print('throughput: (inferences per second)', samples / recorded_time)
