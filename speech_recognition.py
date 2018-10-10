# Property of RickSaitamaSanchez
# Comandos sublime:
# Mover uma linha(s): ctrl + shift + seta (para cima ou baixo)
# Comentar linha(s): Selecione as linhas + ctrl + /
# Identar: para frente ( ctrl + ] ); para trás ( ctrl + [ )
# Para desativar os warnings do python: python3 -W ignore speech_recognition.py

# -*- coding: utf-8 -*- 

def printline():
	print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

def lstm_model(existing_model=False, model_name="", usage="train", lstm1_n=100, lstm2_n=100):

	if (usage == 'train' and existing_model==False):
		name = input("Name of the model: ")
	else:
		name = model_name

	print("Importing libraries...")
	import numpy as np
	import tflearn
	# TFlearn is a modular and transparent deep learning library built on top of Tensorflow. 
	# It was designed to provide a higher-level API to TensorFlow in order to facilitate and 
	# speed-up experimentations, while remaining fully transparent and compatible with it.
	import speech_data # speech_data will fetch data from web and format it for us.
	import matplotlib.pyplot as plt
	print("Imported libraries!")

	# Hyperparameters: 
	# are the variables which determines the network structure (Eg: Number of Hidden Units)
	# and the variables which determine how the network is trained (Eg: Learning Rate).
	learning_rate = 0.001 # Original is 0.0001
	# Learning rate is a hyper-parameter that controls how much we are adjusting the weights 
	# of our network with respect the loss gradient. The greater the learning rate the faster
	# our network trains, the lower the learning rate the more accurate our network predicts.
	training_iters = 100 # (Original is 30000)
	# Since spoken are a sequence of sound waves, we should use a recurrent neural network
	# because of its ability to process sequences. Lets build it below:
	width = 20  # mfcc features
	height = 80  # (max) length of utterance
	classes = 10 # amout of targets for using in layers
	split_p = 0.9 # split percentage (Size of training set). Testing set will be 1 - split_p
	dataset = 2400 # Size of dataset
	batchsize = int(split_p*dataset) # Used batch for training
	lstm1_neurons = lstm1_n # Amout of lstm neurons
	lstm2_neurons = lstm2_n
	dropout = 0.8 # amout of dropout (disabling neurons during training)
	print("Hyperparameters were set!")


	print("Loading batch...")
	if(usage == 'train'):
		batch = speech_data.mfcc_batch_generator(dataset)
		# This function (mfcc_batch_generator(batch_size)) will download (if needed)
		# a set of WAV files with recordings of spoken digits and a label with that digit. Having
		# the files, it will randomly load the batchs (with .wav files and their respective labels)
		# Original batch_size: 64
		print("Loading training and testing sets...")
		wav_files, wav_labels = next(batch) # Spliting files and its labels with python built-in next() function.
		trainX, trainY = wav_files[:batchsize], wav_labels[:batchsize] # Training set gets firsts 90% of dataset
		testX, testY = wav_files[batchsize:], wav_labels[batchsize:]  # Validation set gets lasts 10% of dataset
		print("Training and testing sets were loaded!")
	elif(usage == 'test'):
		batch = speech_data.mfcc_batch_generator(200)
		testX, testY = next(batch)

	# Overfitting refers to a model that models the “training data” too well. Overfitting happens
	# when a model learns the detail and noise in the training data to the extent that it 
	# negatively impacts the performance of the model on new data.

	# Loading or building model:
	print("Building/Loading neural network structures...")
	net = tflearn.input_data([None, width, height]) 
	# The input_data is a layer that will be used as the input layer.
	# For example, if the network wants an input with the shape [None, img_size,img_size,1] 
	# meaning in human language:
	# None - many or a number of or how many images of.(batch size)
	# img_size X img_size - dimensions of the image.
	# 1 - with one color channel.
	net = tflearn.lstm(net, lstm1_neurons, dropout=dropout, return_seq=True) # First parameter is net1, since we are feeding
	# tensors from one layer to the next. 128 means the number of neurons, too few would lead to
	# bad predictions, and to many would overfit the net. The third parameter, dropout, says how
	# much dropout do we want. Droupout helps prevent overfiting by randomly turning off some
	# neurons during training, so data is forced to find new paths between layers, allowing for
	# a more generalized model.
	# lstm is a type of RNN that can remember everything that is fed, outperforming regular
	# recurrent neural networks.
	net = tflearn.lstm(net, lstm2_neurons)  
	net = tflearn.fully_connected(net, classes, activation='softmax') # The activation function
	# softmax will convert numerical data into probabilities.
	net = tflearn.regression(net, optimizer='adam', learning_rate=learning_rate, loss='categorical_crossentropy')
	# The output layer is a regression, which will output a single predicted number for our utterance.
	# The adam optimizer minimize the categorical cross entropy loss function over time.
	model = tflearn.DNN(net, tensorboard_verbose=3, tensorboard_dir='Graphs')

	# if(existing_model==True): model.load('tflearn.lstm.model_' + model_name) # Load weights, if wanted.
	print("Built net!")

	# Training and saving model:
	if(usage=="train"):
		print("\nStarting the training!")
		for i in range(int(training_iters)): # Each iteration have 10 training epochs.
			treino = model.fit(trainX, trainY, n_epoch=10, validation_set=(testX, testY), show_metric=True, batch_size=batchsize, run_id=name)	
		print("Network has been successfully trained!")

		model.save('tflearn.lstm.model_' + name)
		print("Model saved with name: tflearn.lstm.model_" + name)

	# Printing predictions:	
	if(usage=="test"):
		_Y = model.predict(testX)
		print("\nPredictions:")
		printline()
		accuracy = 0
		for i in range(len(testX)):
			# if(i < int(len(validation_labels)*split_p)):
			# 	print("Training sample")
			# else: print("Validation sample")
			prediction = []
			target = []
			for j in range(len(_Y[i])):
				prediction.append(str(round(_Y[i][j]*100, 1)) + "%") # Making predictions readable
				target.append('Nope' if testY[i][j]==0 else ' Yes ') # Making targets readable
			if(_Y[i].tolist().index(max(_Y[i])) == testY[i].tolist().index(max(testY[i]))): accuracy += 1
			print("Prediction " + str(i+1) + ":", prediction) # Predição da rede treinada (Cada lista contém a probabilidade de cada número falado (classe))
			print("Target " + str(i+1) + "    :", target) # Targets
			printline()		
		accuracy /= len(testX)
		print("TEST ACCURACY: %.1f%%" % (accuracy*100))
		printline()

	# Saving weight matrices:

	# W_fc = np.matrix(model.get_weights(net3.W))
	# W_reg = np.matrix(model.get_weights(net4.W))
	# np.savetxt('Weight_matrix_fully_connected', W_fc)
	# np.savetxt('Weight_matrix_regression', W_reg)
	# print("\nSuccessfully saved layers weights matrices!\n")
