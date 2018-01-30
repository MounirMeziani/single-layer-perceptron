import numpy as np 

class mlp(object):

	def __init__(self, num_inputs, num_outputs, learning_rate):
		self._num_inputs = num_inputs
		self._num_outputs = num_outputs
		self._c = learning_rate

		#self.hidden_layer = np.random.randint(3, size=(self._hidden_size, self._num_inputs + 1))
		self.output_layer = np.random.randint(3, size=(self._num_outputs, self._num_inputs + 1))

	def _activation(self, z):
		return float(1 / (1 + np.exp(-z))) #sigmoid activation
	
	def eval(self, x):
		"""
		:param x:<list> 1 x m+1 feature vector
		"""
		print(x)
		output = np.dot(self.output_layer, x.T)
		output = map(output, self._activation)

		#final_output = np.dot(self.output_layer, hidden_output)
		#final_output = np.map(final_output, self._activation)

		return np.argmax(output)

	def train(self, train_x, labels, num_epochs=1):

		accuracy_log = []

		for epoch in range(num_epochs):
			results = []

			for i, x in enumerate(train_x):
				y = self.eval(x)
				results.append(y)

				#adjust weights in each layer
				sign = np.sign(labels[i] - y)
				self.output_layer += sign * self._c * x

			accuracy = accuracy_score(results, labels)
			print(f'Epoch {epoch} - Training Accuracy: {accuracy}\n')
			accuracy_log.append(accuracy)

		return accuracy_log

	def test(self, test_x):

		test_y = map(test_x, self.eval)

		return test_y