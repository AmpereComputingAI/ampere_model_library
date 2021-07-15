from transformers import TFBertForSequenceClassification
from transformers import BertTokenizer, glue_convert_examples_to_features
import tensorflow_datasets as tfds
from transformers import BertForSequenceClassification
from transformers import TFBertForSequenceClassification, TFTrainer, TFTrainingArguments
from sklearn.metrics import precision_recall_fscore_support
import tensorflow as tf

# Models can also be trained natively in TensorFlow 2. Just as with PyTorch, TensorFlow models can be instantiated with
# from_pretrained() to load the weights of the encoder from a pretrained model.

model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased')

# Let’s use tensorflow_datasets to load in the MRPC dataset from GLUE. We can then use our built-in
# glue_convert_examples_to_features() to tokenize MRPC and convert it to a TensorFlow Dataset object.
# Note that tokenizers are framework-agnostic, so there is no need to prepend TF to the pretrained tokenizer name.
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
data = tfds.load('glue/mrpc')
train_dataset = glue_convert_examples_to_features(data['train'], tokenizer, max_length=128, task='mrpc')
train_dataset = train_dataset.shuffle(100).batch(32).repeat(2)

# The model can then be compiled and trained as any Keras model:

optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer=optimizer, loss=loss)
model.fit(train_dataset, epochs=2, steps_per_epoch=115)

# With the tight interoperability between TensorFlow and PyTorch models, you can even save the model and then reload it
# as a PyTorch model (or vice-versa):
model.save_pretrained('./my_mrpc_model/')
pytorch_model = BertForSequenceClassification.from_pretrained('./my_mrpc_model/', from_tf=True)

# We also provide a simple but feature-complete training and evaluation interface through Trainer() and TFTrainer().
# You can train, fine-tune, and evaluate any 🤗 Transformers model with a wide range of training options and with
# built-in features like logging, gradient accumulation, and mixed precision.

model = TFBertForSequenceClassification.from_pretrained("bert-large-uncased")

training_args = TFTrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=3,              # total # of training epochs
    per_device_train_batch_size=16,  # batch size per device during training
    per_device_eval_batch_size=64,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
)

trainer = TFTrainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=tfds_train_dataset,    # tensorflow_datasets training dataset
    eval_dataset=tfds_test_dataset       # tensorflow_datasets evaluation dataset
)

# Now simply call trainer.train() to train and trainer.evaluate() to evaluate. You can use your own module as well,
# but the first argument returned from forward must be the loss which you wish to optimize.

# Trainer() uses a built-in default function to collate batches and prepare them to be fed into the model.
# If needed, you can also use the data_collator argument to pass your own collator function which takes in the data in
# the format provided by your dataset and returns a batch ready to be fed into the model. Note that TFTrainer() expects
# the passed datasets to be dataset objects from tensorflow_datasets.

# To calculate additional metrics in addition to the loss, you can also define your own compute_metrics
# function and pass it to the trainer


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