#!/usr/bin/env bash

set -e

# Make sure we are using up to date Python 3.7
sudo apt-get install python3.7-tk -y
sudo apt-get install --only-upgrade python3.7-tk -y

# Install ta-lib requirements
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
tar -xzf ta-lib-0.4.0-src.tar.gz
cd ta-lib/
./configure --prefix=/usr
make
sudo make install
cd ..

# Install other packages
pip install -r requirements.txt
pip install -r requirements-dev.txt

