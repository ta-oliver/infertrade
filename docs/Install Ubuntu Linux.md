<p align="center">
  <img src="https://www.infertrade.com/static/media/InferTradeLogo.5c2cc437.svg" alt="InferTrade"/>
</p>




# How to Install Infertrade Package for Ubuntu Linux.


 
- Install [`Python 3.7.x`](https://www.python.org/).
```
sudo apt install python3.7
``` 

- Install [`pip`](https://pip.pypa.io/en/stable/) package manager. 
 
 ```
 sudo apt install python3-pip
 ```

- Install [`TA-Lib`](https://www.ta-lib.org/) technical analysis package for Linux.
    ```
    sem-version python 3.7
    wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
    tar -xzf ta-lib-0.4.0-src.tar.gz
    cd ta-lib/
    ./configure --prefix=/usr
    make
    sudo make install
    cd ..
    pip3 install -r requirements.txt
    pip3 install -r requirements-dev.txt
    ```
 
- Install [`infertrade`](https://github.com/ta-oliver/infertrade) package using [`pip`](https://pip.pypa.io/en/stable/) command in `terminal`.
 ```
 pip3 install infertrade
 ```
- Installation should run without errors, it should install all [`dependencies`](https://github.com/ta-oliver/infertrade/blob/holderfolyf-patch-1/requirements.txt) required.
- For advanced users, add [`TA-Lib`](https://www.ta-lib.org/) to the installation.
```
pip3 install infertrade[ta-lib]
```
- Ensure all [`dependencies`](https://github.com/ta-oliver/infertrade/blob/holderfolyf-patch-1/requirements.txt) are installed by running.
 ```
 pip3 list
 ```
## Testing using PyTest

  - Navigate to [`infertrade-main/tests`](https://github.com/ta-oliver/infertrade/tree/main/tests) directory using `terminal`  and run [`pytest`](https://pytest.org/en/stable/) command.
  ```
  pytest
  ```
  - To run specific tests add filename after [`pytest`](https://pytest.org/en/stable/) command. Replace `test_filename.py` with your required filename.
  ```
  pytest test_filename.py
  ```
 
## Alternative Installation Method from source.
 
- Clone Repo. 
  - Using `terminal`.
  ```
  git clone https://github.com/ta-oliver/infertrade.git
  ```
  - Alternative method: [clone](https://github.com/ta-oliver/infertrade/tree/main) source from website.
- Change directory to [`infertrade`](https://github.com/ta-oliver/infertrade).
 
 ```
 cd infertrade
 ```
- Run [`setup.py`](https://github.com/ta-oliver/infertrade/blob/main/setup.py) file.
 ```
 python3 setup.py install
 ```
- Installation should complete successfully without errors.
