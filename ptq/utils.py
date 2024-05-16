from colorama import Fore, Style,init


init(autoreset=True)
def print_r(text,*args):
    print(Fore.RED + text,*args)
def print_g(text,*args):
    print(Fore.GREEN + text,*args)
def print_y(text):
    print(Fore.YELLOW + text)
def print_b(text):
    print(Fore.BLUE + text)
def print_m(text):
    print(Fore.MAGENTA + text)
def print_c(text):
    print(Fore.CYAN + text)

def show_model_params(model):
    for name, param in model.named_parameters():
        print_b(f'name: {name}, param: {param.data}')

def show_model_struct(model):
    print_g(f'{model}:\n ')

def show_model_ouput(output):
    print_m(f'{output}:\n ')