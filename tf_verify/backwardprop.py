import numpy as np
import collections
import copy


class VectorVar:
    # W is weights matrix, b is bias vector, dim is dimensions of vector
    def __init__(self, name, dim, W=None, b=None, cf=1):
        self.name = name
        self.dim = dim
        if W is None:
            self.W = np.identity(dim)
        else:
            self.W = W
        if b is None:
            self.b = np.zeros((dim))
        else:
            self.b = b
        self.cf = cf


class Term:
    # vars is a list of VectorVars, cfs is the list of corresponding coefficients, and isAbs is whether term is in absolute value
    # caseConds is set of case conditions, which are triples, first element is W, second element is var, third element is b
    def __init__(self, name, vars, isAbs=False, cf=1, caseConds=[]):
        self.varArray = vars
        self.vars = {var.name: var for var in vars}
        self.cf = cf
        self.isAbs = isAbs
        self.caseConds = caseConds
        self.name = name

    def copy(self):
        vars = [VectorVar(var.name, var.dim, np.copy(var.W), np.copy(var.b), var.cf) for var in self.varArray]
        return Term(self.name, vars, self.isAbs, self.cf, copy.deepcopy(self.caseConds))


class Condition:
    def __init__(self, terms):
        # self.terms = terms
        self.termNameMap = {term.name: term for term in terms}


def condsToString(conds):
    for disj, cond in enumerate(conds):
        s = ''
        if disj != 0:
            s += '\nv '

        termConds = ''

        for i, term in enumerate(cond.termNameMap.values()):
            if term.cf == 0:
                break

            if i != 0:
                s += ' + '

            if term.cf != 1:
                s += str(term.cf)

            if term.isAbs:
                s += '|'
            else:
                s += '('

            for j, var in enumerate(term.vars):
                if j != 0:
                    s += ' + '
                if term.vars[var].cf != 1:
                    s += str(term.vars[var].cf)
                s += '('
                s += str(term.vars[var].W) + term.vars[var].name + ' + ' + str(term.vars[var].b)
                # s += 'W' + term.vars[var].name + ' + b'

                s += ')'

            if term.isAbs:
                s += '|'
            else:
                s += ')'

            for varCond in term.caseConds:
                termConds += ' ^ ('

                for l, c in enumerate(varCond):
                    if l != 0 and l % 3 != 1:
                        termConds += ' + '
                    termConds += str(c)
                # if varCond[l] < 0:
                # 	termConds += '-'+ varCond[0] + '_' + str((varCond[l]+1)*-1)
                # else:
                # 	termConds += varCond[0] + '_' + str(varCond[l]-1)

                termConds += ' >= 0)'

        s += ' >= 0 '
        s += termConds
        print(s)


# def matmul(conds, termName, var, W):
# 	for cond in conds:
# 		cond.termNameMap[termName].vars[var].W = cond.termNameMap[termName].vars[var].W.dot(W)
# 		cond.termNameMap[termName].vars[var].dim = cond.termNameMap[termName].vars[var].W.shape[0]


def matmulTerm(conds, termName, W):
    for cond in conds:
        for var in cond.termNameMap[termName].vars:
            cond.termNameMap[termName].vars[var].W = cond.termNameMap[termName].vars[var].W.dot(W)
            cond.termNameMap[termName].vars[var].dim = cond.termNameMap[termName].vars[var].W.shape[0]
            for i in range(len(cond.termNameMap[termName].caseConds)):
                if cond.termNameMap[termName].caseConds[i][1] == var:
                    cond.termNameMap[termName].caseConds[i][0] = cond.termNameMap[termName].caseConds[i][0].dot(W)
        # this would be a place where turning caseConds into a map with var as key would be more efficient


# def biasAdd(conds, termName, var, b):
# 	for cond in conds:
# 		cond.termNameMap[termName].vars[var].b = cond.termNameMap[termName].vars[var].b + cond.termNameMap[termName].vars[var].W.dot(b)

def biasAddTerm(conds, termName, b):
    for cond in conds:
        for var in cond.termNameMap[termName].vars:
            cond.termNameMap[termName].vars[var].b = cond.termNameMap[termName].vars[var].b + \
                                                     cond.termNameMap[termName].vars[var].W.dot(b)
            for i in range(len(cond.termNameMap[termName].caseConds)):
                if cond.termNameMap[termName].caseConds[i][1] == var:
                    cond.termNameMap[termName].caseConds[i][2] = cond.termNameMap[termName].caseConds[i][2] + \
                                                                 cond.termNameMap[termName].caseConds[i][0].dot(b)

        # this would be a place where turning caseConds into a map with var as key would be more efficient


# comp is component of vectorVar to apply relu to
def relu(conds, termName, var, comp):
    for i in range(len(conds)):
        cond = conds.pop()

        # I think this can be made more efficient by just reusing the condition instead of making copy
        cond1 = Condition(cond.termNameMap.values())
        term1 = cond1.termNameMap[termName].copy()
        cond1.termNameMap[termName] = term1

        cond2 = Condition(cond.termNameMap.values())
        term2 = cond2.termNameMap[termName].copy()
        cond2.termNameMap[termName] = term2

        dim = term1.vars[var].dim
        term1.caseConds.append([np.identity(dim)[comp], var, 0])
        term2.caseConds.append([-1*np.identity(dim)[comp], var, 0])

        term2.vars[var].W[:,comp] = 0

        conds.appendleft(cond2)
        conds.appendleft(cond1)


def reluLayer(conds, termName, var):
    layerSize = conds[-1].termNameMap[termName].vars[var].dim
    for i in range(layerSize):
        relu(conds, termName, var, i)


def reluLayerTerm(conds, termName):
    for var in conds[-1].termNameMap[termName].vars:
        layerSize = conds[-1].termNameMap[termName].vars[var].dim
        for i in range(layerSize):
            relu(conds, termName, var, i)

# def conv2DLayerTerm(conds, termName, stride, kernelSize, filters, convFilters, convShape, inp, out):
#     for cond in conds:
#         for var in cond.termNameMap[termName].vars:
#             W = np.zeros((out[0]*out[1]*out[2]*out[3],inp[0]*inp[1]*inp[2]))
#             for i in range(convShape[-1]):
#                 for j in range(convShape[-2]):
#                     convMatrix = convFilters[:,:,j,i]


