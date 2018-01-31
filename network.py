__author__ = "Ben Pang"
__copyright__ = "2018, Ben Pang"

import numpy as np 
from sklearn.metrics import accuracy_score

class two_layer_perceptron(object):

	def __init__(self, num_inputs, num_outputs, learning_rate, threshold):
		self._num_inputs = num_inputs
		self._num_outputs = num_outputs
		self._c = learning_rate
		self._threshold = threshold


		#Implement the output layer as a matrix, where each node is represented by a row
		#Each row is the weight vector for that node
		self.output_layer = 3 - 6*np.random.random(size=(self._num_outputs, self._num_inputs + 1))

	def _activation(self, z):
		"""
		Heaviside step activation
		:param z:<double> output of weighted sum
		"""
		result = 1 if z >= self._threshold else 0
		return result
	
	def eval(self, x):
		"""
		:param x:<list> 1 x m+1 feature vector
		"""
		output = np.dot(self.output_layer, x.T)
		output = list(map(self._activation, output))

		return np.argmax(output), output

	def train(self, train_x_raw, labels, num_epochs=1):
		"""
		:param train_x_raw:<nparray> training data n x m numpy array
		:param labels: <nparray> label vector n = 1 numpy array
		:param num_epochs:<int> number of passes through the training data before stopping training
		"""
		#append a column of 1's for the bias
		train_x = np.hstack([train_x_raw, np.ones((train_x_raw.shape[0], 1))])
		accuracy_log = []

		for epoch in range(num_epochs):
			results = []

			for i, x in enumerate(train_x):
				y, output = self.eval(x)
				results.append(y)

				for node in range(self._num_outputs):
					if(output[node] == 1 and node != labels[i]):
						self.output_layer[node] -= self._c * x
					elif(output[node] == 0 and node == labels[i]):
						self.output_layer[node] += self._c * x

			accuracy = accuracy_score(results, labels)
			#print(f'Epoch {epoch} - Training Accuracy: {accuracy}\n')
			accuracy_log.append(accuracy)

		return accuracy_log

	def predict(self, test_x_raw):

		test_x = np.hstack([test_x_raw, np.ones((test_x_raw.shape[0], 1))])
		test_y = list(map(lambda x: self.eval(x)[0], test_x))

		return test_y