import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2LMHeadModel, GPT2Tokenizer


tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
# model = GPT2Model.from_pretrained(model_name, torchscript=True)

model = GPT2LMHeadModel.from_pretrained("gpt2", torchscript=True).eval()
text = "Hi, how are you?"
encoded_input = tokenizer.encode(text, return_tensors='pt')

model = torch.jit.trace(model, torch.randint(10000, (5,)))
scripted_model = torch.jit.script(model)
# output = model.generate(encoded_input, max_length=30, num_beams=2, no_repeat_ngram_size=2, early_stopping=True)
output = scripted_model(encoded_input)
# torch_out = scripted_model(context)

print(output)
print(type(output))
print(len(output))
generated_text_torch = tokenizer.decode(output)

# print("Fragment: {}".format(sentence_fragment))
print("Completed: {}".format(generated_text_torch))

# print(output)
# print(type(output))
#
# print(tokenizer.decode(output[0], skip_special_tokens=True))