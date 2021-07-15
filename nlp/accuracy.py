import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizerFast
import tensorflow as tf
from transformers import TFDistilBertForSequenceClassification, TFTrainer, TFTrainingArguments

df = messages = pd.read_csv('/onspecta/dev/krishnaik_transformers/SMSSpamCollection', sep='\t',
                           names=["label", "message"])
print(df.head())

X = list(df['message'])
y = list(df['label'])

y = list(pd.get_dummies(y, drop_first=True)['spam'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

train_encodings = tokenizer(X_train, truncation=True, padding=True)
test_encodings = tokenizer(X_test, truncation=True, padding=True)


train_dataset = tf.data.Dataset.from_tensor_slices((
    dict(train_encodings),
    y_train
))

test_dataset = tf.data.Dataset.from_tensor_slices((
    dict(test_encodings),
    y_test
))

training_args = TFTrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=2,              # total number of training epochs
    per_device_train_batch_size=8,  # batch size per device during training
    per_device_eval_batch_size=16,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=10,
)
with training_args.strategy.scope():
    model = TFDistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")

trainer = TFTrainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,         # training dataset
    eval_dataset=test_dataset             # evaluation dataset
)

# print(trainer.evaluate(test_dataset))
print(trainer.evaluate())
