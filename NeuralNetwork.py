import numpy as np
np.random.seed(1)

class NeuralNetwork():
	
	def __init__(self, layers, learning_rate):

		self.layers = layers
		self.learning_rate = learning_rate
		
		self.inputs_nodes = self.layers[0]
		self.hidden_layer_1_nodes = self.layers[1]
		self.hidden_layer_2_nodes = self.layers[2]
		self.output_nodes = self.layers[3]

		#init weights
		self.w1 = 2 * np.random.random((self.inputs_nodes, self.hidden_layer_1_nodes)) - 1
		self.w2 = 2 * np.random.random((self.hidden_layer_1_nodes, self.hidden_layer_2_nodes)) - 1
		self.w3 = 2 * np.random.random((self.hidden_layer_2_nodes, self.output_nodes)) - 1

		

		#just to save the errors so we can plot them later
		self.errors_recorder = []


	def info(self):
		print "Input Nodes: ", self.inputs_nodes	
		print "Fisrt Hidden Layer Nodes: ", self.hidden_layer_1_nodes	
		print "Second Hidden Layer Nodes: ", self.hidden_layer_2_nodes	
		print "Output Nodes: ", self.output_nodes	
		#print "Total Weights", len(self.w1) + len(self.w2) + len(self.w3)
	
	# activation function
	def sigmoid(self, x):
		return 1.0/(1.0 + np.exp(-x))

	# derivative of activation function
	def sigmoid_prime(self, x):
		return x*(1-x)

	#training
	def train(self, inputs, targets, epochs):
		for epoch in range(epochs):

			#feed forward
			a1 = self.sigmoid(np.dot(inputs, self.w1))
			a2 = self.sigmoid(np.dot(a1, self.w2))
			a3 = self.sigmoid(np.dot(a2, self.w3))

			# start backpropagation

			error_a3 = 0
			error_a2 = 0
			error_a1 = 0

			#error in a3
			error_a3 = targets - a3
			del_a3 = error_a3 * self.sigmoid_prime(a3)

			#display the error
			if(epoch % 1000 == 0):
				epoch_error =  np.mean(np.abs(error_a3))
				self.errors_recorder.append(epoch_error)
				print "Epoch: ", epoch, "error = ",epoch_error
			
			#error in a2
			error_a2 = np.dot(del_a3, self.w3.T)
			del_a2 = error_a2 * self.sigmoid_prime(a2)

			#error in a2
			error_a1 = np.dot(del_a2, self.w2.T)
			del_a1 = error_a1 * self.sigmoid_prime(a1)

			#adjust weight
			self.w3 += self.learning_rate * np.dot(a2.T,del_a3)
			self.w2 += self.learning_rate * np.dot(a1.T,del_a2)
			self.w1 += self.learning_rate * np.dot(inputs.T,del_a1)


	
	def predect(self, inputs):
		a1 = self.sigmoid(np.dot(inputs, self.w1))
		a2 = self.sigmoid(np.dot(a1, self.w2))
		a3 = self.sigmoid(np.dot(a2, self.w3))
		return a3

	
	#red data from file as array
	@staticmethod
	def read_data(file_path, separator):
		results = []
		file = open(file_path, "r")
		for line in file:
			line = line.rstrip('\n') 
			arr = line.split(separator)
			results.append([float(x) for x in arr])
		file.close()
		return results
	
