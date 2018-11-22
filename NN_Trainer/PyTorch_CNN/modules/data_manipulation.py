import torchvision # Have data loaders for common datasets such as Imagenet, CIFAR10, MNIST, etc. And data transformers
import torchvision.transforms as transforms
import torch
import texts
import os
import auxiliary as aux
from PIL import Image

def load_trainset_CIFAR10(batch_size=4):
	transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
	trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform) # 50000 images with shape 32x32x3
	trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True) # Will shuffle even in training
	classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
	return trainloader, classes

def load_testset_CIFAR10(batch_size=4):
	transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
	testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform) # 10000 test images
	testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True)
	classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
	return testloader, classes

def load_model(net):
	texts.print_blox("CARREGAR")
	path = 'architectures'
	# print("PATH: ", os.getcwd())
	print("Arquivos encontrados:\n")
	files = os.listdir(path)
	for i in files:
		if (not i.endswith('~')):
			aux.printc("lyellow", i)
	name = input("\nInsira o nome do arquivo: ")
	path = path + '/' + name
	if(any(name == i for i in files)):
		net.load_state_dict(torch.load(path))
		print("Sucesso no carregamento!\n")
	else:
		aux.printc("red", "O arquivo "+path+" não foi encontrado!\n")
		aux.press_enter()

def save_model(net):
	texts.print_blox("SALVAR")
	name = input("Insira o nome com o qual deseja salvar: ")
	path = 'architectures/' + name
	torch.save(net.state_dict(), path)
	print("Salvo com sucesso!\n")

def reset_weights(net):
	net.load_state_dict(torch.load('architectures/reset'))

def load_image(path):
	image = Image.open(path)
	transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
	image = transform(image).float()
	image = image.unsqueeze(0)
	return image