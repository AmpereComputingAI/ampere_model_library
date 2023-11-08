import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
# model = GPT2Model.from_pretrained(model_name, torchscript=True)

model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-6B", pad_token_id=tokenizer.eos_token_id)
text = "Hi, how are you?"
encoded_input = tokenizer.encode(text, return_tensors='pt')
model.eval()
torch.jit.script(model)
output = model.generate(encoded_input, max_length=30, num_beams=2, no_repeat_ngram_size=2, early_stopping=True)

print(output)
print(type(output))

print(tokenizer.decode(output[0], skip_special_tokens=True))