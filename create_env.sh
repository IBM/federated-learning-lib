#!/usr/bin/env bash
#Author: Mark Purcell (markpurcell@ie.ibm.com)

#Might need this if automating to avoid typimg 'yes' to prompt
#ssh-keyscan -t rsa github.com >> ~/.ssh/known_hosts

echo "Installing virtual environment..."
python3 -m pip install --user virtualenv

echo "Creating virtual environment..."
virtualenv venv
source venv/bin/activate

echo "Upgrading pip..."
python3 -m pip install --upgrade pip

echo "Please run 'source venv/bin/activate' to enable the virtual environment"
