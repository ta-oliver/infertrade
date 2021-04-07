**How to Install Infertrade Package from scratch for Ubuntu.**

Linux installation steps:

```
sem-version python 3.7
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
tar -xzf ta-lib-0.4.0-src.tar.gz
cd ta-lib/
./configure --prefix=/usr
make
sudo make install
cd ..
pip install -r requirements.txt
pip install -r requirements-dev.txt
```


