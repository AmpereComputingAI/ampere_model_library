import pandas as pd
import argparse
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizerFast
import tensorflow as tf
from transformers import TFDistilBertForSequenceClassification, TFTrainer, TFTrainingArguments
from sklearn.metrics import precision_recall_fscore_support
from seqeval.metrics import accuracy_score, f1_score, precision_score, recall_score
import wget
import subprocess
from downloads.utils import get_downloads_path
import pathlib
from pathlib import Path
import os

dataset_filename = 'SMSSpamCollection'
dataset_url = 'https://www.dropbox.com/s/ymd54rur6atkvqu/SMSSpamCollection'

model_filename = 'model.tar.gz'
model_url = 'https://www.dropbox.com/s/fp43347je178wo8/model.tar.gz'


downloads_dir_path = pathlib.Path(get_downloads_path())

path_to_dataset = Path(os.path.join(downloads_dir_path, dataset_filename))
path_to_model = Path(os.path.join(downloads_dir_path, 'senti_model'))

if path_to_dataset.is_file():
    pass
else:
    try:
        subprocess.run(["wget", dataset_url])
        subprocess.run(["mv", dataset_filename, str(downloads_dir_path)])
    except KeyboardInterrupt:
        subprocess.run(["rm", dataset_filename])

if path_to_model.is_dir():
    pass
else:
    try:
        subprocess.run(["wget", model_url])
        subprocess.run(["tar", "-xf", model_filename, "-C", str(downloads_dir_path)])
        subprocess.run(["rm", model_filename])
    except KeyboardInterrupt:
        subprocess.run(["rm", model_filename])


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


def test_accuracy(path_to_model, path_to_dataset):

    df = pd.read_csv(path_to_dataset, sep='\t',
                               names=["label", "message"])

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
        model = TFDistilBertForSequenceClassification.from_pretrained(path_to_model)

    trainer = TFTrainer(
        model=model,                         # the instantiated 🤗 Transformers model to be trained
        args=training_args,                  # training arguments, defined above
        train_dataset=train_dataset,         # training dataset
        eval_dataset=test_dataset,             # evaluation dataset
        compute_metrics=compute_metrics
    )

    accuracy = trainer.evaluate()

    print('\n')
    print("EVAL ACCURACY: ")
    print((accuracy['eval_accuracy']))


test_accuracy(path_to_model, path_to_dataset)
