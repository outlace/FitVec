# FitVec.py
A simple numpy-based genetic algorithm library for function parameter optimization.

#### Summary

You can use FitVec to "evolve" a vector of parameters that optimizes a function.
This was originally made specifically for optimizing the weights of a neural network,
however, it should be useful in a broad spectrum of optimization problems.

#### Installation
There's no installer yet, but it's a single Python function in a single Python file so for now
just include the file in your project directory and import it.

#### How to use

The library consists of a single function, `evolveParams`, that expects a
function to minimize (e.g. a cost function), the vector length,some optional hyperparameters and any
additional parameters the supplied cost function needs (e.g. training data).
The vector length is the size of the vector of parameters you are trying to evolve.
For example, if you're using this for a neural network, you will need to construct
a 1-dimensional vector out of all your weight matrices. If the total number of weights is
say 10, then the vector length parameter will be 10. It will return an optimized set of weights
in the form of a 10-length numpy array.

The optional hyperparameters tune the function of the genetic algorithm.
These are the initial population size, mutation rate (eg 0.01 (1%) ), and the number of generations (e.g. 250; also known as epochs).
These are supplied in the form of a tuple, e.g. (100, 0.01, 250).
These default to (100, 0.01, 100) when not explicitly supplied.
A mutation rate of 1% (0.01), in general, should not be changed as it empirically seems to be a good value.
The initial population size and the number of generations should be changed to match the complexity of your problem.
For a small feedforward neural network, an initial population of 100 and run for less than 50 generations will likely work fine.
As a rule of thumb, the initial population size should be approximately 10 times the length of your parameter vector.

Here's the function definition:
`evolveParams(costFunction, vecLength, params=(100,0.01,100), *args)`

The last parameter of your cost function must be the parameter vector, because
it is called in this form: `costFunction(*args, optimal_params)`

#### Example using an XOR neural network

```python
import numpy as np
import math
import FitVec as fv

X = np.array([[0,0,1],[0,1,1],[1,0,1],[1,1,1]])
y = np.array([[0, 1, 1, 0]]).T
init_theta = 10*(np.random.random((13,1)) - 0.5)

def runForward(X, theta):
	theta1 = np.array(theta[:9]).reshape(3,3)
	theta2 = np.array(theta[9:]).reshape(4,1)
	h1 = sigmoid(np.dot(X, theta1))
	h1_bias = np.insert(h1, 3, [1,1,1,1], axis=1)
	output = sigmoid(np.dot(h1_bias, theta2))
	return output

def costFunction(X, y, theta):
	m = float(len(X))
	hThetaX = np.array(runForward(X, theta))
	return np.sum(np.abs(y - hThetaX))
def sigmoid(x): return 1 / (1 + np.exp(- x))

def demoRun():
	print("Random theta: \n%s\n" % (np.round(runForward(X, init_theta), 2)))
	print("Cost: %s\n" % (costFunction(X,y, init_theta)))
	optimal_theta = fv.evolveParams(costFunction, 13, (100,0.01,25), X, y)
	print("Optimal theta: \n%s\n" % (np.round(runForward(X, optimal_theta.reshape(13,1)), 2)))
	print("Cost: %s\n" % (costFunction(X, y, optimal_theta.reshape(13,1))))
demoRun()
```

Output:
```
Random theta:
[[ 0.61]
 [ 0.72]
 [ 0.33]
 [ 0.46]]

Cost: 2.01846904619

Best Sol'n:
[[  4.67190922  12.22       -24.91378912 -14.84       -11.67        11.3712786
   -2.99       -10.17        -6.98         7.39        44.52292934
   17.58023122  -6.48      ]]
Cost:0.00377452533293
Optimal theta:
[[ 0.]
 [ 1.]
 [ 1.]
 [ 0.]]

Cost: 0.00377452533293
```
