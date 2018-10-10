import matplotlib.pyplot as plt
from matplotlib import style
import csv

def loss_graph(file_name, graph_type):
	path = "csv/" + file_name + ".csv"
	x = []
	y = []
	with open(path, 'r') as csvfile:
		plots = csv.reader(csvfile, delimiter=',')
		next(plots, None)
		for row in plots:
			x.append(int(row[1]))
			y.append(round(float(row[2]), 4))
	plt.plot(x,y,label=graph_type)
	plt.xlabel('Épocas')
	if(graph_type=='validation_accuracy'):
		plt.title('Acurácia de validade')
		plt.ylabel('Acurácia')
	elif(graph_type=='train_accuracy'):
		plt.title('Acurácia de treino')
		plt.ylabel('Acurácia')
	elif(graph_type=='train_loss'):
		plt.title('Perda da etapa de treino')
		plt.ylabel('Erro')
	elif(graph_type=='validation_loss'):
		plt.title('Perda da etapa de validação')
		plt.ylabel('Erro')

	plt.legend()
	plt.show()

loss_graph("Arch5_val_acc", "validation_accuracy")
loss_graph("Arch5_train_acc", "train_accuracy")
loss_graph("Arch5_train_loss", "train_loss")
loss_graph("Arch5_val_loss", "validation_loss")