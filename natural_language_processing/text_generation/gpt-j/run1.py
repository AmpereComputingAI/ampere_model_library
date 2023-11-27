import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2", torchscript=True)

# model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-6B", pad_token_id=tokenizer.eos_token_id, torchscript=True)
text = "Hi, how are you?"
encoded_input = tokenizer(text, return_tensors='pt')
# print(encoded_input)
input_dict = {key: value for key, value in encoded_input.items()}

def model_wrapper(input_ids):
    return model(input_ids)[0]

# traced_model = torch.jit.trace(model, (input_dict['input_ids'],))
traced_model = torch.jit.trace(model_wrapper, encoded_input)

frozen_model = torch.jit.freeze(traced_model)

#output = frozen_model(**encoded_input)

with torch.no_grad():
    output = frozen_model(encoded_input)

decoded_output = tokenizer.decode(output.argmax(dim=-1)[0])

# output_ids = output.logits.argmax(dim=-1)
# decoded_output = tokenizer.decode(output_ids[0])
#
# tokenizer.decode(output, skip_special_tokens=True)
