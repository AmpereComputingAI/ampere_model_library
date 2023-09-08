from transformers import GPT2Tokenizer, GPT2Model

model = GPT2Model.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

text = "Replace me by any text you'd like."
encoded_input = tokenizer(text, return_tensors='pt')


output = model(**encoded_input)

