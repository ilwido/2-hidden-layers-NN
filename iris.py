import numpy as np
import matplotlib.pyplot as plt
import NeuralNetwork as nn


irisNN = nn.NeuralNetwork([4,5,4,3], learning_rate = 0.01) 


all_data = nn.NeuralNetwork.read_data(file_path="iris_training_dataset.data", separator=",")

all_data = np.array(all_data)
inputs = all_data[:,:4]
output = all_data[:,-1]

final_outputs = []
for i in output:
	j = np.zeros(3) + 0.01
	j[int(i)] = 0.99
	final_outputs.append(j)

final_outputs = np.array(final_outputs, ndmin=2)

#scale units :)
inputs =  inputs / np.amax(inputs, axis=0)
final_outputs =  final_outputs / np.amax(final_outputs, axis=0)

#train the neural network
irisNN.train(inputs, final_outputs, 20000)


#test the neural network
all_data_test = nn.NeuralNetwork.read_data(file_path="iris_testing_dataset.data", separator=",")
all_data_test = np.array(all_data_test)

inputs_test = all_data_test[:,:4]
output_test = all_data_test[:,-1]

inputs_test =  inputs_test / np.amax(inputs_test, axis=0)


predections = irisNN.predect(inputs_test)


scorecard = []
for i in range(len(predections)):
	target = output_test[i]
	predect= np.argmax(predections[i])
	if(target == predect):
		scorecard.append(1)
	else:
		scorecard.append(0)


scorecard_array = np.asarray(scorecard)
performance =  float(scorecard_array.sum()) / float(scorecard_array.size)
print "Performance = ", performance * 100, "%"

#plot the eroros
plt.plot(irisNN.errors_recorder)
plt.show()