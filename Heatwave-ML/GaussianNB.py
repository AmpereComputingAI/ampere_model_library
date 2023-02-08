from sklearn.naive_bayes import GaussianNB
from skl2onnx import convert_sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from skl2onnx.common.data_types import FloatTensorType, Int64TensorType
#create an instance of model
model=GaussianNB()
#skl2onnx.convert_sklearn(model)
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2,  shuffle=True, random_state=123)
model.fit(X_train, y_train)
initial_type = [('Input_X', FloatTensorType(None,X_train.shape[1]))]
onx = convert_sklearn(model,initial_types=initial_type)
with open("GaussianNB.onnx", "wb") as f:
    f.write(onx.SerializeToString())

import onnxruntime as rt
import numpy as np
sess = rt.InferenceSession("GaussianNB.onnx",providers=['AIOExecutionProvider'])
from time import time
input_name = sess.get_inputs()[0].name
label_name = sess.get_outputs()[0].name
runtime=[]
niter=100000
for i in range(niter):
    start = time()
    pred_onx = sess.run([label_name], {input_name: X_test.astype(np.float32)})[0]
    runtime.append(time() - start)
runtime=np.median(runtime)
print(pred_onx)
print(np.mean(pred_onx == y_test))
print('Throughput:', X_test.shape[0]/runtime)
print('Latency [ms]:', runtime/X_test.shape[0]*1000)
"""
throughput how many samples to process in a sec
latency is how many sec is taking to process one sample

given 150 samples.
"""
