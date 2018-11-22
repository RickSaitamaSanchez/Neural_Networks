import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision # Have data loaders for common datasets such as Imagenet, CIFAR10, MNIST, etc. And data transformers
import torchvision.transforms as transforms
from summary import summary
import data_manipulation as dm
import auxiliary as aux
import progressBar
import texts

def train(net, epochs, trainset_size=12500, batch_size=4, learning_rate=0.001):
	texts.print_blox("TREINO")
	# TRAINING:
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
	trainloader, classes = dm.load_trainset_CIFAR10(batch_size)
	print("Training features:")
	print("Images: %i" % (trainset_size*batch_size))
	print("Mini-batch size: %i" % (batch_size))
	print("Learning rate: %.3f" % (learning_rate))
	print("Epochs: %i" % (epochs))
	print("")
	for epoch in range(epochs): # loop over the dataset multiple times
		running_loss = 0.0
		acc = 0.0
		for i, data in enumerate(trainloader, 0):
			inputs, labels = data # get the inputs
			optimizer.zero_grad() #zero the parameter gradients
			# forward + backward + optimize
			outputs = net(inputs)
			# print("Labels:", labels)
			# print("outputs:", outputs)
			acc += (outputs.max(dim=1)[1] == labels).sum()
			loss = criterion(outputs, labels)
			loss.backward()
			optimizer.step()
			# print statistics
			running_loss += loss.item() # Returns the value of this tensor as a standard Python number. This only works for tensors with one element. 
			progressBar.printProgressBar(i, trainset_size-1, prefix = 'Epoch '+str(epoch+1)+':', suffix = 'Complete', length = 50)
			if i == trainset_size-1:
				print("Report: ", end='')
				print('[Loss: %.3f; Hits: %i; Acc: %.1f%%]' % ( running_loss/trainset_size, acc, float(acc)/(1.0*trainset_size*batch_size)*100.0 ) )
				break
	print("Finished training!\n")

def test(net, testset_size=2500, batch_size=4):
	texts.print_blox("TESTE")
	testloader, classes = dm.load_testset_CIFAR10(batch_size)
	correct = 0
	total = 0
	class_correct = list(0. for i in range(10))
	class_total = list(0. for i in range(10))
	with torch.no_grad():
		for i, data in enumerate(testloader, 0):
			images, labels = data
			outputs = net(images)
			_, predicted = torch.max(outputs.data, 1)
			total += labels.size(0)
			correct += (predicted == labels).sum().item()
			c = (predicted == labels).squeeze()
			if(batch_size>1):
				for j in range(batch_size):
					label = labels[j]
					class_correct[label] += c[j].item()
					class_total[label] += 1
			else:
				label = labels[0]
				class_correct[label] += c
				class_total[label] += 1
			if(i == testset_size-1):
				break
	print('Accuracy of the network on the '+str(testset_size*batch_size)+' test images: %d%%' % (100*correct/total))
	for i in range(10):
		print('Accuracy of %5s: %2d%%' % (
			classes[i], 100 * class_correct[i] / class_total[i]))
	print()

def imshow(img):
	img = img / 2 + 0.5 # unnormalize
	npimg = img.numpy()
	plt.imshow(np.transpose(npimg, (1, 2, 0)))

def pred(net, batch_size):
	option = input("Deseja escolher um arquivo? (s/n): ")
	if(option == 's' or option == 'S'):
		path = 'Images'
		print("Arquivos encontrados:\n")
		files = os.listdir(path)
		for i in files:
			if (not i.endswith('~')):
				aux.printc("lyellow", i)
		name = input("\nInsira o nome do arquivo: ")
		path = path + '/' + name
		if(any(name == i for i in files)):
			image = dm.load_image(path)
			classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
			output = net(image)
			_, predicted = torch.max(output, 1)
			print("Predição da rede: " + classes[predicted])
			print()
		else:
			aux.printc("red", "O arquivo "+path+" não foi encontrado!")
			print()
	else:
		testloader, classes = dm.load_testset_CIFAR10(batch_size)
		dataiter = iter(testloader)
		images, labels = dataiter.next()
		imshow(torchvision.utils.make_grid(images))
		print('Labels das imagens: ', ' '.join('%5s' % classes[labels[j]] for j in range(batch_size)))
		plt.title('Amostra de teste')
		plt.show()
		outputs = net(images)
		_, predicted = torch.max(outputs, 1)
		print("Predição da rede: ", ' '.join('%5s' % classes[predicted[j]] for j in range(batch_size)))
		print()

def print_model(net, batch_size):
	aux.printc("blue", disable=False)
	summary(net, (3, 32, 32), batch_size)
	aux.printc(disable=True)
