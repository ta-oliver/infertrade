<p align="center">
  <img src="https://www.infertrade.com/static/media/InferTradeLogo.5c2cc437.svg" alt="InferTrade"/>
</p>




# How to Install Infertrade Package for Microsoft Windows 10.


 
- Install [`Python 3.7.x`](https://www.python.org/).
 
- Add Path to Environment Variables.

 ![alt-text](https://github.com/ta-oliver/infertrade/blob/main/docs/images/run_dialog_box.jpg)
 ![alt-text](https://github.com/ta-oliver/infertrade/blob/main/docs/images/2%20environment_variables.jpg)
 ![alt-text](https://github.com/ta-oliver/infertrade/blob/main/docs/images/3edit_environment_variables.jpg)
 ![alt-text](https://github.com/ta-oliver/infertrade/blob/main/docs/images/4add_path.jpg)
 
- Install [`pip`](https://pip.pypa.io/en/stable/) package manager. 
 
  - Download file from  `cmd prompt`:
  ```
  curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
  ```
  - Alternative method: [download](https://bootstrap.pypa.io/get-pip.py) using web browser.
  
  - Navigate to directory using `cmd prompt` and run following command.
  ```
  py get-pip.py
  ```
- Add Path to Environment Variables for [`pip`](https://pip.pypa.io/en/stable/) package manager using `cmd prompt`.
```
setx PATH “%PATH%;C:\Python37\Scripts”
```
- Install [`TA-Lib`](https://www.ta-lib.org/) technical analysis package for windows.
    - Download [whl](https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib) file for your version of [`python`](https://www.python.org/).
    - Run [`pip`](https://pip.pypa.io/en/stable/) install command in `cmd prompt`.
        ```
        pip install TA_Lib-0.4.17-cp37-cp37m-win_amd64.whl
        ```
    - Replace `cp37` with correct [`python`](https://www.python.org/) version and `win_amd64` with your system architecture in above command.
 
- Install [`infertrade`](https://github.com/ta-oliver/infertrade) package using [`pip`](https://pip.pypa.io/en/stable/) command in `cmd prompt`.
 ```
 pip install infertrade
 ```
- Installation should run without errors, it should install all [`dependencies`](https://github.com/ta-oliver/infertrade/blob/holderfolyf-patch-1/requirements.txt) required.
- For advanced users, add [`TA-Lib`](https://www.ta-lib.org/) to the installation.
```
pip3 install infertrade[ta-lib]
```
- Ensure all [`dependencies`](https://github.com/ta-oliver/infertrade/blob/holderfolyf-patch-1/requirements.txt) are installed by running.
 ```
 pip list
 ```
## Testing using PyTest

  - Navigate to [`infertrade-main/tests`](https://github.com/ta-oliver/infertrade/tree/main/tests) directory using `cmd prompt`  and run [`pytest`](https://pytest.org/en/stable/) command.
  ```
  pytest
  ```
  - To run specific tests add filename after [`pytest`](https://pytest.org/en/stable/) command. Replace `test_filename.py` with your required filename.
  ```
  pytest test_filename.py
  ```
 
## Alternative Installation Method from source.
 
- Clone Repo. 
  - Using `cmd prompt`.
  ```
  git clone https://github.com/ta-oliver/infertrade.git
  ```
  - Alternative method: [clone](https://github.com/ta-oliver/infertrade/tree/main) source from website.
- Change directory to [`infertrade`](https://github.com/ta-oliver/infertrade).
 
 ```
 cd infertrade
 ```
- Run [`setup`](https://github.com/ta-oliver/infertrade/blob/main/setup.py) file.
 ```
 python setup.py install
 ```
- Alternate command:
 ```
 py setup.py install
 ```
- Installation should complete successfully without errors.
