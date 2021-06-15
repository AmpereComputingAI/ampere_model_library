import gzip
import json
from transformers import TensorFlowBenchmark, TensorFlowBenchmarkArguments, BertConfig
import tensorflow as tf
from tensorflow.python.profiler import model_analyzer
from datetime import datetime


args = TensorFlowBenchmarkArguments(models=["bert-base-uncased"], batch_sizes=[8], sequence_lengths=[8])

benchmark = TensorFlowBenchmark(args)
results = benchmark.run()


with gzip.open("../logs/plugins/profile/2021_06_15_15_34_43/49524a0fc1dd.trace.json.gz", "r") as f:
    data = f.read()
    j = json.loads(data.decode('utf-8'))
    print(j)


tf.DLS.print_profile_data()

print(results)

