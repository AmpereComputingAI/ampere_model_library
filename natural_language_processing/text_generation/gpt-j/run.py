from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import GPT2LMHeadModel, GPT2Tokenizer

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-6B")
# model = GPT2LMHeadModel.from_pretrained('gpt2', pad_token_id=tokenizer.eos_token_id)

print(tokenizer.decode(tokenizer.eos_token_id))

sentence = "hey, how are you?"
input_ids = tokenizer.encode(sentence, return_tensors='pt')
print(tokenizer.decode(input_ids[0], skip_special_tokens=True))
quit()
#
# print(input_ids)
# print(type(input_ids))
# quit()

print('1')
output = model.generate(input_ids, max_length=30, num_beams=2, no_repeat_ngram_size=2, early_stopping=True)

print(output)
print(type(output))

print(tokenizer.decode(output[0], skip_special_tokens=True))