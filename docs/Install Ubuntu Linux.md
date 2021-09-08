<p align="center">
  <img src="https://www.infertrade.com/static/media/InferTradeLogo.5c2cc437.svg" alt="InferTrade"/>
</p>

# How to Install Infertrade Package for Ubuntu Linux.

- Install [`Python 3.7.x`](https://www.python.org/).

  ```
  sudo apt install python3.7
  ``` 

- Installing Python 3.7 on newer Ubuntu systems.

  Python 3.7 is not among the packages maintained by Ubuntu after 18.04. If regular installation fails, there are two methods to install it:
  - Install from a PPA
  - Compile from source

  It is recommended that you do the install from the PPA first:
  ```
  sudo apt update
  sudo apt install software-properties-common
  sudo add-apt-repository ppa:deadsnakes/ppa
  sudo apt-get install python3.7-tk
  ```

  If you get SSL errors while using pip, you will have to install from source. Thankfully, this is not an unreasonable task. It is detailed Josh Spicer's blog post: [SSL issues with Python 3.7 Install From Source](https://joshspicer.com/python37-ssl-issue).


- Install [`pip`](https://pip.pypa.io/en/stable/) package manager. 
 
  ```
  sudo apt install python3-pip
  ```
 
- Install [`infertrade`](https://github.com/ta-oliver/infertrade) package using [`pip`](https://pip.pypa.io/en/stable/) command in `terminal`.
  ```
  pip3 install infertrade
  ```
- Installation should run without errors, it should install all [`dependencies`](https://github.com/ta-oliver/infertrade/blob/holderfolyf-patch-1/requirements.txt) required.
 
- Ensure all [`dependencies`](https://github.com/ta-oliver/infertrade/blob/holderfolyf-patch-1/requirements.txt) are installed by running.
  ```
  pip3 list
  ```
<br>

## When Python 3.7 is not your latest version.


- To install infertrade specifically for the required version of python (v3.7).
  ```
  python3.7 -m pip install infertrade
  ```

- Your python program may give you a dependency related error when running Infertade module or not be loading other relevant modules like pandas, even after installing them with pip3. 

  In this case you want to use the same procedure as above.
  ```
  python3.7 -m pip install pandas
  ```
<br>

## Testing using PyTest.


  - Navigate to [`infertrade-main/tests`](https://github.com/ta-oliver/infertrade/tree/main/tests) directory using `terminal`  and run [`pytest`](https://pytest.org/en/stable/) command.

    ```
    pytest
    ```
  - To run specific tests add filename after [`pytest`](https://pytest.org/en/stable/) command. Replace `test_filename.py` with your required filename.

    ```
    pytest test_filename.py
    ```
 <br>

# Alternative Installation Method from source.
 

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
