import torch
import time
from transformers import GPT2Tokenizer, GPT2Model

model = GPT2Model.from_pretrained('gpt2', torchscript=True)
model.eval()
model = torch.jit.freeze(torch.jit.script(model))


tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

text = "Replace me by any text you'd like."
encoded_input = tokenizer(text, return_tensors='pt')

traced_model = torch.jit.trace(model, (encoded_input,))
frozen_model = torch.jit.freeze(traced_model)

with torch.no_grad():
    start = time.time()
    output = model.generate(encoded_input, max_length=50, num_return_sequences=1)
    finish = time.time()
    print(finish - start)



# import torch
# from transformers import GPT2LMHeadModel, GPT2Tokenizer
#
# def generate_text(prompt, model, tokenizer):
#     # Encode the prompt text to tensor
#     input_ids = tokenizer.encode(prompt, return_tensors="pt")
#
#     # Generate text given the prompt
#     with torch.no_grad():
#         output = model.generate(input_ids, max_length=50, num_return_sequences=1)
#
#     # Decode the generated text back to a string
#     generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
#
#     return generated_text
#
# if __name__ == "__main__":
#     # Load pre-trained GPT-2 model and tokenizer
#     model = GPT2LMHeadModel.from_pretrained("gpt2")
#     tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
#
#     # Create example input
#     example_input = tokenizer.encode("Once upon a time", return_tensors="pt")
#
#     # Trace the model
#     traced_model = torch.jit.trace(model, (example_input,))
#
#     # Freeze the model
#     frozen_model = torch.jit.freeze(traced_model)
#
#     # Generate text
#     prompt = "Once upon a time"
#     generated_text = generate_text(prompt, frozen_model, tokenizer)
#     print(f"Generated text:\n{generated_text}")