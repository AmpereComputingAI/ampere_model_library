from transformers import AutoTokenizer, GPT2LMHeadModel, AutoModelForCausalLM
import torch
import time
import os
torch.set_num_threads(int(os.environ["AIO_NUM_THREADS"]))

tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2", torchscript=True)

#tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6b")
#model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-6b")

model.eval()
inputs = tokenizer.encode("Hello, I'm looking for an employment, ", return_tensors="pt")
print("\nNo tracing\n")
with torch.no_grad():
    for n in range(3):
        break
        x = time.time()
        outputs = model.generate(inputs, do_sample=True, max_length=100, top_k=50, top_p=0.95, num_return_sequences=1)
        print(f"Run: {n}, throughput: {round(outputs.shape[1] / (time.time() - x), 3)} tps")


#model.forward = torch.jit.freeze(torch.jit.trace_module(model, {"forward": inputs}))
model.generate = torch.jit.freeze(torch.jit.trace_module(model, {"generate": inputs}))

print("\nTracing engaged\n")
# with torch.no_grad():
#     for n in range(3):
#         x = time.time()
#         outputs = model.generate(inputs, do_sample=True, max_length=100, top_k=50, top_p=0.95, num_return_sequences=1)
#         print(f"Run: {n}, throughput: {round(outputs.shape[1] / (time.time() - x), 3)} tps")
# text = tokenizer.decode(outputs[0], skip_special_tokens=True)
# print(text)


with torch.no_grad():
    x = time.time()
    outputs = model.generate(inputs, do_sample=True, max_length=50, top_p=0.95)
    print(f"throughput: {round(outputs.shape[1] / (time.time() - x), 3)} tps")
text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(text)
