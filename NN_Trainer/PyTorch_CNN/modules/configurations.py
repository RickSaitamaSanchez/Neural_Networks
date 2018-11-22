import os
import auxiliary as aux
from auxiliary import printc
import texts
import fileinput
import sys

def get_configs():
	path = 'configurations/configs'
	if(os.path.isfile(path)):
		with open(path, 'r') as f:
			text = f.read()
			matrix = [item.split() for item in text.split('\n')[:-1]]
			# print(matrix)
			learning_rate = float(matrix[2][1][1:])
			minibatch = int(matrix[3][1][1:])
			trainset = int(matrix[4][1][1:])
			testset = int(matrix[5][1][1:])
			epochs = int(matrix[6][1][1:])
			return learning_rate, minibatch, trainset, testset, epochs
	else:
		aux.printc("red","O arquivo "+path+" está corrompido ou não existe!")
		aux.press_enter()
		return 0.001, 4, 1250, 250, 2

def replaceAll(file,searchExp,replaceExp):
    for line in fileinput.input(file, inplace=1):
        if searchExp in line:
            line = replaceExp
        sys.stdout.write(line)

def set_configs(config, value):
	path = 'configurations/configs'
	if(not os.path.isfile(path)):
		aux.printc("red","O arquivo "+path+" está corrompido ou não existe!")
		aux.press_enter()
		return
	if(config=='learning_rate'): replaceAll(path, 'learning_rate', 'learning_rate          ='+str(value)+'\n')
	elif(config=='minibatch'): replaceAll(path, 'minibatch', 'minibatch              ='+str(value)+'\n')
	elif(config=='trainset'): replaceAll(path, 'trainset', 'trainset               ='+str(value)+'\n')
	elif(config=='testset'): replaceAll(path, 'testset', 'testset                ='+str(value)+'\n')
	elif(config=='epochs'): replaceAll(path, 'epochs', 'epochs                 ='+str(value)+'\n')
	else:
		aux.printc("red", "Configuração inválida!")
		aux.press_enter()
		return


def options():
	learning_rate, minibatch, trainset, testset, epochs = get_configs()
	printc("cyan","1. Learning rate: \033[1;33m" + str(learning_rate))
	printc("cyan","2. Minibatch: \033[1;33m" + str(minibatch))
	printc("cyan","3. Conjunto de treino: \033[1;33m" + str(trainset))
	printc("cyan","4. Conjunto de teste: \033[1;33m" + str(testset))
	printc("cyan","5. Épocas: \033[1;33m" + str(epochs))
	printc("cyan","6. Voltar.\n")
	option = input("Insira a opção desejada: ")
	aux.clear_screen()
	if(option=='1'):
		x = input("Insira o novo valor: ")
		set_configs('learning_rate', x)
		menu()
	elif(option=='2'):
		x = input("Insira o novo valor: ")
		set_configs('minibatch', x)
		menu()
	elif(option=='3'):
		x = input("Insira o novo valor: ")
		set_configs('trainset', x)
		menu()
	elif(option=='4'):
		x = input("Insira o novo valor: ")
		set_configs('testset', x)
		menu()
	elif(option=='5'):
		x = input("Insira o novo valor: ")
		set_configs('epochs', x)
		menu()
	elif(option=='6'):
		return
	else:
		menu(True)
def menu(error_message=False):
	aux.clear_screen()
	texts.print_blox("CONFIGURAÇÕES")
	if(error_message):
		aux.error_option()
	options()
