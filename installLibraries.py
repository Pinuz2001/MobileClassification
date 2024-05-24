import subprocess
import sys


def install_packages(requirements_file='requirements.txt'):
    try:
        with open(requirements_file, 'r') as file:
            packages = file.readlines()
    except FileNotFoundError:
        print(f"Il file {requirements_file} non è stato trovato.")
        return

    for package in packages:
        package = package.strip()
        if package:
            print(f"Installazione del pacchetto: {package}")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])


if __name__ == "__main__":
    install_packages()
