from transformers import BertConfig, BertModel

# Building the config
config = BertConfig()

# Building the model from the config
model = BertModel(config)

print(config)

model.save_pretrained("/Users/tanvir.aumi/Desktop/work/nlp-starter/local")