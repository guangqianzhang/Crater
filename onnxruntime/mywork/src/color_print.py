

from colorama import Fore, Back, Style, init

# 初始化 Colorama，以便在所有平台上都能工作
init(autoreset=True)

def print_color(msg, color):
    print(f"{getattr(Fore, color)}{msg}{Style.RESET_ALL}")

def print_success(msg, color="GREEN"):
    print(f"{getattr(Fore, color)}{msg}{Style.RESET_ALL}")
    
def print_info(msg, color="GREEN"):
    print(f"{getattr(Fore, color)}{msg}{Style.RESET_ALL}")

def print_error(msg, color="RED"):
    print(f"{getattr(Fore, color)}{msg}{Style.RESET_ALL}")

def print_warning(msg, color="YELLOW"):
    print(f"{getattr(Fore, color)}{msg}{Style.RESET_ALL}")


