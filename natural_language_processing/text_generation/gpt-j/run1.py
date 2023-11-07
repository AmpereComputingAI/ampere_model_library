import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
# model = GPT2Model.from_pretrained(model_name, torchscript=True)

model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-6B", pad_token_id=tokenizer.eos_token_id, torchscript=True)
text = "Hi, how are you?"
encoded_input = tokenizer(text, return_tensors='pt')
print(encoded_input)
input_dict = {key: value for key, value in encoded_input.items()}
print(input_dict)
quit()

traced_model = torch.jit.trace(model, (input_dict['input_ids'],))
#traced_model = torch.jit.trace(model, (encoded_input,))
frozen_model = torch.jit.freeze(traced_model)

#output = frozen_model(**encoded_input)
output = frozen_model(input_dict['input_ids'])
