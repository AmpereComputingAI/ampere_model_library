from sklearn.linear_model import LinearRegression
from skl2onnx import convert_sklearn
from sklearn.datasets import load_diabetes
from skl2onnx.common.data_types import FloatTensorType, Int64TensorType
from sklearn.model_selection import train_test_split
#create an instance of model
model=LinearRegression()
#X is set of features and y is corresponding label
X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2,  shuffle=True, random_state=123)
model.fit(X_train, y_train)
initial_type = [('Input_X', FloatTensorType(None,X.shape[1]))]
onx = convert_sklearn(model,initial_types=initial_type)
with open("linear_regression_model.onnx", "wb") as f:
    f.write(onx.SerializeToString())


import onnxruntime as rt
import numpy as np
from sklearn.metrics import r2_score

batchsize = 2048

np.random.seed(123)
indx = np.random.choice(X_test.shape[0], size=batchsize, replace=True)
#print(indx)
#print(X_test.shape)
X_test = X_test[indx]
y_test = y_test[indx]
#print(X_test.shape)
sess = rt.InferenceSession("linear_regression_model.onnx",providers=['AIOExecutionProvider'])
from time import time
#print(sess.get_inputs())
input_name = sess.get_inputs()[0].name
label_name = sess.get_outputs()[0].name
#print(X_test.shape)
runtime = []
niter = 100000
for i in range(niter):
    start = time()
    pred_onx = sess.run([label_name], {input_name: X_test.astype(np.float32)})[0]
    runtime.append(time() - start)
runtime = np.median(runtime)
#print(pred_onx)
#print('RMSE:', np.mean((pred_onx - y_test)**2)**0.5)
#print('R2:', r2_score(y_test, pred_onx))
print('Throughput- num of samples per sec:', X_test.shape[0]/runtime)
print('Latency [us]:', runtime/X_test.shape[0]*1e6)

