import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2', pad_token_id=tokenizer.eos_token_id)

# print(tokenizer.decode(tokenizer.eos_token_id))

sentence = "hey, how are you?"
input_ids = tokenizer.encode(sentence, return_tensors='pt')

# Trace the model
traced_model = torch.jit.trace(model, torch.randint(10000, (5,)))

# Freeze the model
frozen_model = torch.jit.freeze(traced_model)

with torch.no_grad():
    outputs = frozen_model(input_ids)

# Decode the output
output_ids = outputs.logits.argmax(dim=-1)
decoded_output = tokenizer.decode(output_ids[0])

print(decoded_output)
