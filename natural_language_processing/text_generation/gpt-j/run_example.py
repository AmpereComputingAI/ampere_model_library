from transformers import AutoTokenizer, GPT2LMHeadModel, AutoModelForCausalLM
import torch
import time
import os
torch.set_num_threads(int(os.environ["AIO_NUM_THREADS"]))

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
# model = GPT2LMHeadModel.from_pretrained("gpt2", torchscript=True)
model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-6B", pad_token_id=tokenizer.eos_token_id, torchscript=True).eval()

#tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6b")
#model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-6b")

model.eval()
inputs = tokenizer.encode("Hello, I'm looking for an employment, ", return_tensors="pt")
print("\nNo tracing\n")
# with torch.no_grad():
#     for n in range(1):
#         break
#         x = time.time()
#         outputs = model.generate(inputs, do_sample=True, max_length=100, top_k=50, top_p=0.95, num_return_sequences=1)
#         print(f"Run: {n}, throughput: {round(outputs.shape[1] / (time.time() - x), 3)} tps")


#model.forward = torch.jit.freeze(torch.jit.trace_module(model, {"forward": inputs}))
model.generate = torch.jit.freeze(torch.jit.trace_module(model, {"generate": inputs}))

print("\nTracing engaged\n")
with torch.no_grad():
    x = time.time()
    outputs = model.generate(inputs, do_sample=True, max_length=100, top_k=50, top_p=0.95, num_return_sequences=1)
    print(f"throughput: {round(outputs.shape[1] / (time.time() - x), 3)} tps")
text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(text)




# TORCH_COMPILE=1 AIO_NUM_THREADS=80 python natural_language_processing/text_generation/gpt/run-v2-compile.py -m gpt2 --lambada_path /ampere/aml/lambada_test_plain_text.txt --num_runs 4

# AIO_NUM_THREADS=80 python natural_language_processing/text_generation/gpt/run_example3.py
# AIO_NUM_THREADS=80 python natural_language_processing/text_generation/gpt/run-v2-compile.py -m gpt2 --lambada_path /ampere/aml/lambada_test_plain_text.txt --num_runs 4