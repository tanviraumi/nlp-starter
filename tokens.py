from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

sequence = "Using a Transformer network is simple"

print(tokenizer(sequence))

tokens = tokenizer.tokenize(sequence)

print(tokens)

ids = tokenizer.convert_tokens_to_ids(tokens)

print(ids)

decoded_string = tokenizer.decode(ids)

print(decoded_string)