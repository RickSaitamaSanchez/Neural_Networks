import torch
import actions
import models
import data_manipulation as dm
import auxiliary as aux
from auxiliary import printc
import texts
import configurations as config
import os

os.chdir('modules')

net = models.CNN()

def options():
	learning_rate, minibatch, trainset, testset, epochs = config.get_configs()
	printc("cyan","1. Visualizar arquitetura.")
	printc("cyan","2. Treinar rede.")
	printc("cyan","3. Testar rede.")
	printc("cyan","4. Predição individual.")
	printc("cyan","5. Carregar modelo.")
	printc("cyan","6. Salvar modelo.")
	printc("cyan","7. Resetar pesos.")
	printc("cyan","8. Configurações.")
	printc("cyan","9. Créditos.")
	printc("cyan","10. Sair.\n")
	option = input("Insira a opção desejada: ")
	aux.clear_screen()
	if(option=='1'):
		actions.print_model(net, minibatch)
		aux.press_enter()
		menu()
	elif(option=='2'):
		actions.train(net, epochs, trainset, minibatch, learning_rate) # (net, epochs, trainset, minibatch, learning rate)
		aux.press_enter()
		menu()
	elif(option=='3'):
		actions.test(net, testset, minibatch) # (net, testset, minibatch)
		aux.press_enter()
		menu()
	elif(option=='4'):
		actions.pred(net, minibatch)
		aux.press_enter()
		menu()
	elif(option=='5'):
		dm.load_model(net)
		menu()
	elif(option=='6'):
		dm.save_model(net)
		aux.press_enter()
		menu()
	elif(option=='7'):
		dm.reset_weights(net)
		menu()
	elif(option=='8'):
		config.menu()
		menu()
	elif(option=='9'):
		texts.credit()
		aux.press_enter()
		menu()
	elif(option=='10'):
		aux.clear_screen()
		exit(0)
	else:
		menu(True)

def menu(error_message=False):
	aux.clear_screen()
	texts.print_blox()
	if(error_message):
		aux.error_option()
	options()

if __name__ == "__menu__":
	menu()