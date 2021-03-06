from tensorflow_translator import *
from onnx_translator import *
from read_net_file import *
import tensorflow as tf
from backwardprop import *
import collections

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

model, _, means, stds = read_tensorflow_net('mnist_conv_maxpool.tf', 784,
                                            False)  # first param is filename, second is num of pixels, third is trained with pytorch

translator = TFTranslator(model, False)  # second param indicates model is not onnx
operations, resources = translator.translate()
# print(resources[7]['deeppoly'])



# for i in range(len(operations)):
#     print(resources[i]['deeppoly'])
#     print(operations[i])

# print(operations[::-1])

# W = np.array([[[[1,2]],[[0,5]]],[[[-1,2]],[[2,7]]]])
# x = np.array([[[0],[4],[2],[1]],[[-1],[0],[1],[-2]],[[3],[1],[2],[0]],[[0],[1],[4],[1]]])
# xdim=[4,4,1]
# ydim=[1,2,2,2]
# stride = [2,2]
# conv2DLayerTerm(stride, W, xdim, ydim)
# conv2DLayerTerm(resources[1]['deeppoly'][2], resources[1]['deeppoly'][0], resources[1]['deeppoly'][1], resources[1]['deeppoly'][-1])

inputDim = [4,4,1]
poolDim = [2,2]
maxpoolLayerTerm(poolDim, inputDim)


# -------------------------------------------------------------------------------------------------------------------
# conds = collections.deque([])
# y1 = VectorVar(name='y1', dim=2)
# y2 = VectorVar(name='y2', dim=2, cf=-1)
# term1 = Term(name='y', vars=[y1, y2], isAbs=True, cf=-1)
# x1 = VectorVar(name='x1', dim=1)
# x2 = VectorVar(name='x2', dim=1, cf=-1)
# term2 = Term(name='x', vars=[x1, x2], isAbs=True, cf='k')
# cond = Condition([term1, term2])
# conds.appendleft(cond)
#
# # conds contains the post condition of the entire NN
#
# print('Relu:')  # first backward transformation
# reluLayerTerm(conds, 'y')
#
# print('BiasAdd:')
#
# dummy = np.array([1, 3])
# biasAddTerm(conds, 'y', dummy)
# # biasAddTerm(conds, 'y', resources[-2]['deeppoly'][0])
# # print(condsToString(conds))
# # print(len(conds))
#
# print('Matmul:')
# dummy = np.array([[4, 2], [2, 9]])
# matmulTerm(conds, 'y', dummy)
# # matmulTerm(conds, 'y', resources[-3]['deeppoly'][0])
# print('DONE')
# # condsToString(conds)
# # print(len(conds))
#
# # note that the variables in the cases represent y_i = W_i * y_i + b_i
