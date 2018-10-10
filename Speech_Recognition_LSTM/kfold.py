# Property of RickSaitamaSanchez
# -*- coding: utf-8 -*-

def fprintline(file):
	file.write("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")

def printline():
	print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

def press_enter():
	input("Press enter to continue...")

def clear_screen():
	os.system('cls' if os.name == 'nt' else 'clear')

# def lstm_model_kfold():
def lstm_model_kfold(name, lstm1_n, lstm2_n, fc3_n):

	# Results file manipulation:
	# name = input("Name of the model for future saving: ")
	file = open(name + "_results.txt", "w")
	file.write("Long Short-Therm Memory Recurrent Neural Network for speech recognition.\n")
	file.write("Training with K-Fold.\n")
	file.write("Network name: " + name + ".\n")
	file.write("Owner: Gabriel Furtado Lins Melo.\n\n")

	print("Importing libraries...")
	import numpy as np
	import tflearn
	import speech_data 
	import os
	import time
	import datetime
	print("Imported libraries!")

	# Hyperparameters: 
	learning_rate = 0.001 
	lstm1_neurons = lstm1_n 
	lstm2_neurons = lstm2_n
	fc3_neurons = fc3_n
	# lstm_neurons = 128 
	dropout = 0.8
	activation = 'softmax'
	optimizer = 'adam'
	loss_func = 'categorical_crossentropy'
	
	# Training Features:
	kfold_k = 5
	training_iters = 10 # Multiply by 10 to know how many epochs

	# Dataset Features:	
	dataset = 2400
	split_p = 0.9 
	batchsize = int(split_p*dataset) # 2160
	val_sets = int(batchsize/kfold_k) # 432
	classes = 10 
	width = 20  
	height = 80 
	print("Hyperparameters were set!")

	# Loading batchs and defining dataset / testing set:
	print("Loading batch...")
	batch = speech_data.mfcc_batch_generator(dataset)
	wav_files, wav_labels = next(batch)
	print("Loading training and testing sets...")
	testX, testY = wav_files[batchsize:], wav_labels[batchsize:] # Testing set.
	datasetX, datasetY = wav_files[:batchsize], wav_labels[:batchsize] # Dataset.

	# Loading or building model:
	print("Building/Loading neural network structures...")
	net = tflearn.input_data([None, width, height]) 
	net = tflearn.lstm(net, lstm1_neurons, dropout=dropout, return_seq=True)
	net = tflearn.lstm(net, lstm2_neurons)
	net = tflearn.fully_connected(net, fc3_neurons)
	net = tflearn.fully_connected(net, classes, activation=activation) # Output layer
	net = tflearn.regression(net, optimizer=optimizer, learning_rate=learning_rate, loss=loss_func)
	model = tflearn.DNN(net, tensorboard_verbose=3, tensorboard_dir='Graphs')
	print("Built net!")

	# Validation with K-Fold:
	trainX = []
	trainY = []
	validationX = []
	validationY = []
	validation_accuracy = 0

	# Saving the initial weights for restarting them in k-fold
	model.save('tflearn.lstm.kfold')

	# Results file manipulation:
	fprintline(file)
	file.write("\nNetwork Layers:\n\n")
	file.write("	Input data (size: " + str(width) + " x " + str(height) + ")\n")
	file.write("	Lstm layer 1 (neurons: " + str(lstm1_neurons) + "), (dropout: " + str(dropout) + ")\n")
	file.write("	Lstm layer 2 (neurons: " + str(lstm2_neurons) + "), (dropout: None)\n")
	file.write("	Fully connected 1 (neurons: " + str(fc3_neurons) + ")\n")
	file.write("	Fully connected 2 (output neurons: " + str(classes) + "), (activation: " + str(activation) + ")\n")
	file.write("	Regression layer (optimizer: " + str(optimizer) + "), (loss function: " + str(loss_func) + "), (learning rate: " + str(learning_rate) + ")\n")
	file.write("\nDataset Features:\n\n")
	file.write("	Dataset: " + str(dataset) + "\n")
	file.write("	Batch size used in training: " + str(batchsize) + "\n")
	file.write("	Amount of test files: " + str(dataset-batchsize) + "\n")
	file.write("\nTraining Features:\n\n")
	file.write("	K-Fold \"K\": " + str(kfold_k) + "\n")
	file.write("	Epochs: " + str(training_iters*10) + "\n\n")
	fprintline(file)

	# K-Fold training:
	file.write("\nK-Fold results:\n\n")
	printline()
	print("\nInitiating " + str(kfold_k) + "-Fold training.\n")
	printline()
	kname = "K-Fold_" + name
	start_time = time.time()
	for i in range(kfold_k): 
		# Fixing sets
		validationX, validationY = datasetX[(i*val_sets):((i+1)*val_sets)], datasetY[(i*val_sets):((i+1)*val_sets)]
		trainX, trainY = datasetX.copy(), datasetY.copy()
		trainX[(i*val_sets):((i+1)*val_sets)] = []
		trainY[(i*val_sets):((i+1)*val_sets)] = []
		printline()
		# Fold number printing:
		lista = []
		for j in range(kfold_k):
			lista.append('*VAL.*') if j == i else lista.append('TRAIN')
		print("\nK-Fold \"K\":", (i+1))
		print("Dataset:", lista)
		print()
		printline()
		# Actual training:
		minibatch = batchsize-val_sets # 1728
		for j in range(int(training_iters)): # Each iteration have 10 training epochs.
			model.fit(trainX, trainY, n_epoch=10, validation_set=None, show_metric=False, batch_size=minibatch, run_id=kname)
		# Printing validation accuracy for each fold:
		accuracy = 0
		preds = model.predict(validationX)
		for j in range(len(validationX)):
			if(preds[j].tolist().index(max(preds[j])) == validationY[j].tolist().index(max(validationY[j]))): accuracy += 1
		accuracy /= len(validationX)
		printline()
		print("Fold "+ str(i+1) +" accuracy: %0.1f%%" % (accuracy*100))
		printline()
		file.write("Fold "+ str(i+1) +" accuracy: %0.1f%%\n" % (accuracy*100))
		validation_accuracy += accuracy
		model.load('tflearn.lstm.kfold')
	# Printing mean validation accuracy:
	validation_accuracy /= kfold_k
	printline()
	print("\nValidation phase done!")
	print("Mean validation accuracy: %0.1f%%\n" % (validation_accuracy*100))
	file.write("\nK-Fold validation phase mean accuracy: %0.1f%%\n\n" % (validation_accuracy*100))
	fprintline(file)
	printline()
	print()

	# Final Training phase:
	printline()
	print("\nInitiating final training phase.\n")
	printline()
	file.write("\nFinal training phase results:\n\n")
	# After K-Fold training phase (Using all dataset):
	for j in range(int(training_iters)): # Each iteration have 10 training epochs.
		# model.fit(datasetX, datasetY, n_epoch=10, validation_set=(testX,testY), show_metric=True, batch_size=batchsize, run_id=kname)		
		model.fit(datasetX, datasetY, n_epoch=10, validation_set=None, show_metric=False, batch_size=batchsize, run_id=kname)
	# Printing predictions:	
	printline()
	_Y = model.predict(testX)
	print("\nPredictions using testing set:\n")
	printline()
	hits = 0
	pred_matrix = []
	for i in range(len(testX)):
		prediction = []
		target = []
		for j in range(len(_Y[i])):
			prediction.append(str(round(_Y[i][j]*100, 1)) + "%") # Making predictions readable
			target.append('Nope' if testY[i][j]==0 else ' Yes ') # Making targets readable
		if(_Y[i].tolist().index(max(_Y[i])) == testY[i].tolist().index(max(testY[i]))): hits += 1
		if((i+1) % 10 == 0):
			print("Prediction " + str(i+1) + ":", prediction) # Predição da rede treinada (Cada lista contém a probabilidade de cada número falado (classe))
			print("Target " + str(i+1) + "    :", target) # Targets
			printline()	
	accuracy = hits/len(testX)
	print("Testing set size: %d" % len(testX))
	print("Hits (right predictions): %d" % hits)
	print("Testing accuracy: %0.1f%%" % (accuracy*100))
	file.write("Testing set size: %d\n" % len(testX))
	file.write("Hits (right predictions): %d\n" % hits)
	file.write("Testing accuracy: %0.1f%%\n\n" % (accuracy*100))
	fprintline(file)
	printline()
	end_time = time.time()
	total_time = end_time - start_time
	string_time = time.strftime("%Hh%Mm%Ss", time.gmtime(total_time))
	file.write("\nTotal training time:\n")
	file.write(string_time)
	
	# Saving:
	file.close()
	model.save('tflearn.lstm.model_' + name)
	print("Model saved with name: tflearn.lstm.model_" + name)	
	print("Results saved as: " + name + "_results.txt")	


	# Saving weight matrices:
	# W_fc = np.matrix(model.get_weights(net3.W))
	# W_reg = np.matrix(model.get_weights(net4.W))
	# np.savetxt('Weight_matrix_fully_connected', W_fc)
	# np.savetxt('Weight_matrix_regression', W_reg)
	# print("\nSuccessfully saved layers weights matrices!\n")
