import torch.nn as nn
import torch.nn.functional as F

# Convolution size (same to maxpooling):
# Hout = (Hin + 2 x padding - dilation x (kernel_size - 1) - 1)/stride + 1
# Wout = (Win + 2 x padding - dilation x (kernel_size - 1) - 1)/stride + 1

class CNN(nn.Module): # This means that the class CNN inherits the base class called "nn.Module"
	def __init__(self): # init: kinda constructor. When you call CNN() Python creates an object for you, and passes it as the first parameter to the __init__ method
		super(CNN, self).__init__() # super function can be used to gain access to inherited methods – from a parent or sibling class – that has been overwritten in a class object.
		self.conv1 = nn.Conv2d(3,6,5) # (in_channels, out_channels (n. of filters), kernel_size)
		self.pool = nn.MaxPool2d(2,2) # (kernel_size, stride)
		self.conv2 = nn.Conv2d(6,16,5)
		self.fc1 = nn.Linear(16*5*5,120) # (in_features, out_features)
		self.fc2 = nn.Linear(120,84)
		self.fc3 = nn.Linear(84,10)

	def forward(self, x):
		x = self.pool(F.relu(self.conv1(x)))
		x = self.pool(F.relu(self.conv2(x)))
		x = x.view(-1, 16 * 5 * 5)
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return x
