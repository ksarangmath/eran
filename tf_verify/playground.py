from tensorflow_translator import *
from onnx_translator import *
from read_net_file import *
import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

model, _, means, stds = read_tensorflow_net('mnist_relu_3_50.tf', 784, False) #first param is filename, second is num of pixels, third is trained with pytorch

translator = TFTranslator(model, False)
operations, resources = translator.translate()
print(operations)
print(resources)