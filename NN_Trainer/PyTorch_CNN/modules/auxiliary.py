import os

def clear_screen():
	os.system('cls' if os.name == 'nt' else 'clear')

def press_enter():
	input("\033[1;35mPressione enter para continuar...\033[0m")

def error_option():
	printc("red","Opção não reconhecida!\n")

def printc(color="white", message="", disable=True):
	red = "\033[1;91m"
	lred = "\033[0;91m"
	blue = "\033[0;96m"
	cyan = "\033[1;36m"
	green = "\033[1;32m"
	bgreen = "\033[0;32m"
	yellow = "\033[1;33m"
	lyellow = "\033[0;33m"
	purple = "\033[1;35m"
	white = "\033[0;37m"
	endc = '\033[0m'
	if(color=="red"): color = red
	elif(color=="lred"): color = lred
	elif(color=="blue"): color = blue
	elif(color=="cyan"): color = cyan
	elif(color=="green"): color = green
	elif(color=="bgreen"): color = bgreen
	elif(color=="yellow"): color = yellow
	elif(color=="lyellow"): color = lyellow
	elif(color=="purple"): color = purple
	elif(color=="white"): color = white
	else:
		print("COLOR DOESN'T EXISTS")
		return
	if(disable):
		print(color + message + endc)
	else: print(color + message)

	# para ver as cores e seus números no terminal:
	# 	for i in range(0, 2):
	# ...  for j in range(0, 101):
	# ...   print("\033["+str(i)+";" + str(j) +"m" + str(j) + "\033[0m", end=' ')
	# ... 
