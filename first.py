# First 2 layer Neural Net (3,4,1) units
import numpy as np

def act_func(x, deriv=False):		# using sigmoid
	if deriv == True:
		return x*(1-x)
	return 1/(1+np.exp(-x))

X = np.array([[0,0,1,1],		# dim : n*m where n = number of imput features, m = number of training examples
			[0,1,0,1],
			[1,1,1,1]])

y = np.array([[0,1,1,0]])

np.random.seed(1)

W0 = 0.01*(np.random.random((4,3)))
W1 = 0.01*(np.random.random((1,4)))

alpha = 0.5

for j in range(10000):

	# forward propagation
	a0 = X
	z1 = np.dot(W0, a0)
	a1 = act_func(z1)
	z2 = np.dot(W1, a1)
	a2 = act_func(z2)

	dz2 = a2 - y  						# denotes the error

	if (j % 1000) == 0:
		print('Error', str(np.mean(np.abs(dz2))))

	# Backward propagation

	dW1 = (1/4)*(np.dot(dz2, a1.T))
	dz1 = np.multiply(np.dot(W1.T, dz2), act_func(z1, deriv=True))
	dW0 = (1/4)*(np.dot(dz1, X.T))

#	da2 = (a2 - y)*act_func(a2, deriv = True)
#	da1 = (da2.dot(W1.T))*act_func(a1, deriv = True)

	# update weights : Gradient Descent

	W0 -= alpha*dW0
	W1 -= alpha*dW1
#	W1 -= a1.T.dot(da2)
#	W0 -= a0.T.dot(da1)

	print('Output of training : ')
	print(a2)
