import tensorflow as tf
import tensorflow_datasets as tfds
from sklearn.metrics import precision_recall_fscore_support
from datasets import load_dataset
from datasets import load_metric


from transformers import TFBertForSequenceClassification,  TFTrainer, TFTrainingArguments
from transformers import BertTokenizer, glue_convert_examples_to_features
from transformers import BertForSequenceClassification
from transformers import AutoTokenizer


# ======================================================================================================================
# Models can also be trained natively in TensorFlow 2. Just as with PyTorch, TensorFlow models can be instantiated with
# from_pretrained() to load the weights of the encoder from a pretrained model.
print('here 1')
# ======================================================================================================================

model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased')


# ======================================================================================================================
# Let’s use tensorflow_datasets to load in the MRPC dataset from GLUE. We can then use our built-in
# glue_convert_examples_to_features() to tokenize MRPC and convert it to a TensorFlow Dataset object.
# Note that tokenizers are framework-agnostic, so there is no need to prepend TF to the pretrained tokenizer name.
print('here 2')
# ======================================================================================================================
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
data = tfds.load('glue/mrpc')
train_dataset = glue_convert_examples_to_features(data['train'], tokenizer, max_length=128, task='mrpc')
train_dataset = train_dataset.shuffle(100).batch(32).repeat(2)


# ======================================================================================================================
# The model can then be compiled and trained as any Keras model:
print('here 3')
# ======================================================================================================================

optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer=optimizer, loss=loss)
model.fit(train_dataset, epochs=2, steps_per_epoch=115)


# ======================================================================================================================
# With the tight interoperability between TensorFlow and PyTorch models, you can even save the model and then reload it
# as a PyTorch model (or vice-versa):
print('here 4')
# ======================================================================================================================

model.save_pretrained('./my_mrpc_model/')
pytorch_model = BertForSequenceClassification.from_pretrained('./my_mrpc_model/', from_tf=True)


# ======================================================================================================================
# We also provide a simple but feature-complete training and evaluation interface through Trainer() and TFTrainer().
# You can train, fine-tune, and evaluate any 🤗 Transformers model with a wide range of training options and with
# built-in features like logging, gradient accumulation, and mixed precision.
print('here 5')
# ======================================================================================================================

model = TFBertForSequenceClassification.from_pretrained("bert-large-uncased")


training_args = TFTrainingArguments(
    output_dir='./results',  # output directory
    num_train_epochs=3,  # total # of training epochs
    per_device_train_batch_size=16,  # batch size per device during training
    per_device_eval_batch_size=64,  # batch size for evaluation
    warmup_steps=500,  # number of warmup steps for learning rate scheduler
    weight_decay=0.01,  # strength of weight decay
    logging_dir='./logs',  # directory for storing logs
)

trainer = TFTrainer(
    model=model,  # the instantiated 🤗 Transformers model to be trained
    args=training_args,  # training arguments, defined above
    train_dataset=train_dataset,  # tensorflow_datasets training dataset
    eval_dataset=trainer.eval_dataset  # tensorflow_datasets evaluation dataset
)

trainer.evaluate()

# ======================================================================================================================
# Now simply call trainer.train() to train and trainer.evaluate() to evaluate. You can use your own module as well,
# but the first argument returned from forward must be the loss which you wish to optimize.

# Trainer() uses a built-in default function to collate batches and prepare them to be fed into the model.
# If needed, you can also use the data_collator argument to pass your own collator function which takes in the data in
# the format provided by your dataset and returns a batch ready to be fed into the model. Note that TFTrainer() expects
# the passed datasets to be dataset objects from tensorflow_datasets.

# To calculate additional metrics in addition to the loss, you can also define your own compute_metrics
# function and pass it to the trainer
print('here 6')
# ======================================================================================================================


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


metric = load_metric("accuracy")


# ======================================================================================================================
# DATASETS

# We will use the 🤗 Datasets library to download and preprocess the IMDB datasets. We will go over this part pretty
# quickly. Since the focus of this tutorial is on training, you should refer to the 🤗 Datasets documentation or the
# Preprocessing data tutorial for more information.

# First, we can use the load_dataset function to download and cache the dataset:
# ======================================================================================================================


raw_datasets = load_dataset("imdb")

# ======================================================================================================================
# This works like the from_pretrained method we saw for the models and tokenizers
# (except the cache directory is ~/.cache/huggingface/dataset by default).
# The raw_datasets object is a dictionary with three keys: "train", "test" and "unsupervised" (which correspond to
# the three splits of that dataset). We will use the "train" split for training and the "test" split for validation.

# To preprocess our data, we will need a tokenizer:
# ======================================================================================================================

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")


# ======================================================================================================================
# As we saw in Preprocessing data, we can prepare the text inputs for the model with the following command
# (this is an example, not a command you can execute):
# ======================================================================================================================

inputs = tokenizer(sentences, padding="max_length", truncation=True)

# ======================================================================================================================
# This will make all the samples have the maximum length the model can accept (here 512), either by padding or
# truncating them. However, we can instead apply these preprocessing steps to all the splits of our dataset at once
# by using the map method:
# ======================================================================================================================


def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)

# ======================================================================================================================
# You can learn more about the map method or the other ways to preprocess the data in the 🤗 Datasets documentation.
# Next we will generate a small subset of the training and validation set, to enable faster training:
# ======================================================================================================================

small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))
full_train_dataset = tokenized_datasets["train"]
full_eval_dataset = tokenized_datasets["test"]


