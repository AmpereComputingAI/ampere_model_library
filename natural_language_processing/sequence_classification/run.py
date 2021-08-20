from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
import tensorflow as tf
import pandas as pd
import numpy as np

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased-finetuned-mrpc")
model = TFAutoModelForSequenceClassification.from_pretrained("bert-base-cased-finetuned-mrpc")

# docker
filepath = '/onspecta/dev/mz/datasets/MRPC/msr_paraphrase_test.txt'
# local
# filepath = '/home/marcel/dev/mz/datasets/MRPC/msr_paraphrase_test.txt'

df = pd.read_csv(filepath, sep=r'\t', engine='python')

###################### USEFUL SNIPPET ######################

# df.columns = ['temporary_column']
# df[['Quality', '1_ID', '2_ID', '#1_String', '#2_String']] = df.temporary_column.str.split("\t", expand=True,)
# del df['temporary_column']

# # for i in range(arr_length):
# #      for j in range(3, 5):
# #          print('i: ', i,'j: ', j, sep='')
#            print('the label is: ', arr[i, 0])
# #          print(arr[i,j])
#

###################### WORKING CODE ######################

arr = df.to_numpy()
arr_length = arr.shape[0]
correct = 0
incorrect = 0
count = 0

for i in range(arr_length):
    sequence_0 = arr[i, 3]
    sequence_1 = arr[i, 4]
    label = int(arr[i, 0])

    input = tokenizer(sequence_0, sequence_1, return_tensors="tf")

    paraphrase_classification_logits = model(input)[0]
    result = tf.nn.softmax(paraphrase_classification_logits, axis=1).numpy()[0]

    prediction = np.argmax(result, axis=0)
    if label == prediction:
        correct += 1
    else:
        incorrect += 1
    count += 1
    print('I am working... ', count)

print('correct: ', correct)
print('incorrect: ', incorrect)

