import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load the model and tokenizer
model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Prepare the input
sentence = "Hello, how are you?"
inputs = tokenizer.encode(sentence, return_tensors="pt")

# Define a wrapper function for the model to return only logits (tensors)
def model_wrapper(input_ids):
    return model(input_ids, return_dict=False)[0]

# Trace the model with the wrapper function
traced_model = torch.jit.trace(model_wrapper, inputs)

# Freeze the model
frozen_model = torch.jit.freeze(traced_model)

# Generate output
with torch.no_grad():
    outputs = frozen_model(inputs)

# Decode the output
decoded_output = tokenizer.decode(outputs.argmax(dim=-1)[0])

print(decoded_output)