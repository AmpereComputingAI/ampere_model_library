from transformers import TensorFlowBenchmark, TensorFlowBenchmarkArguments, BertConfig
import tensorflow as tf
from tensorflow.python.profiler import model_analyzer
from datetime import datetime

args = TensorFlowBenchmarkArguments(models=["bert-base-uncased"], batch_sizes=[8], sequence_lengths=[8])

benchmark = TensorFlowBenchmark(args)
results = benchmark.run()

tf.DLS.print_profile_data()

print(results)

