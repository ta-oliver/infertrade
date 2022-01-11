#!/usr/bin/env bash

set -e

# Make sure we are using up to date Python 3.7
sudo apt-get install python3.7-tk -y
sudo apt-get install --only-upgrade python3.7-tk -y

# Install wheel for ta-lib installation dependency
# pip install wheel

# Install other packages
pip install -r requirements.txt
pip install -r requirements-dev.txt
