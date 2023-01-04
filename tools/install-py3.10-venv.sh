#!/bin/sh

# torchvision currently needs python 3.10 while
# on debian there are both 3.11 and 3.10 available

# to run torchvision, python 3.10 virtual environment is required

# apt install python3.10-venv

python3.10 -m venv ~/python-3.10

echo "usage:"
echo "source ~/python-3.10/bin/activate"
