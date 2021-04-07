**How to Install Infertrade Package from scratch for windows.**
 
- Install Python 3.7.x 
 
    [Python Download](https://www.python.org/)
 
- Add Path to Environment Variables

 ![alt-text](https://github.com/ta-oliver/infertrade/blob/main/docs/images/run_dialog_box.jpg)
 ![alt-text](https://github.com/ta-oliver/infertrade/blob/main/docs/images/2%20environment_variables.jpg)
 ![alt-text](https://github.com/ta-oliver/infertrade/blob/main/docs/images/3edit_environment_variables.jpg)
 ![alt-text](https://github.com/ta-oliver/infertrade/blob/main/docs/images/4add_path.jpg)
 
- Install PIP Package Manager 
 
  - Download file from  `cmd prompt`:
  ```
  curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
  ```
  - Alternative method: [download](https://bootstrap.pypa.io/get-pip.py) using web browser
  
  - Navigate to directory in `cmd prompt` and run following command
  ```
  py get-pip.py
  ```
- Add Path to Environment Variables for PIP Package Manager using `cmd prompt`.
```
setx PATH “%PATH%;C:\Python37\Scripts”
```
- Install TA-Lib for windows
    - Download whl file from [whl download](https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib)
    - Run install command in `cmd prompt`.
        ```
        pip install TA_Lib-0.4.17-cp37-cp37m-win_amd64.whl
        ```
    - Replace cp37 with correct python version in above command.
 
- Install infertrade package with pip install command in `cmd prompt`.
 ```
 pip install infertrade
 ```
- Script should run without errors, it will install dependencies as well.
 
- Ensure all dependencies installed by running the following command.
 ```
 pip list
 ```
- Testing using Pytest
  - Navigate to infertrade-main/tests directory using `cmd prompt`  and run `pytest` command.
  ```
  pytest
  ```
  - To run specific tests add filename after `pytest` command. Replace `test_filename.py` with your required filename.
  ```
  pytest test_filename.py
  ```
 
## Alternative Installation Method from source.
 
- Clone Repo. 
  - Using `cmd prompt`.
  ```
  git clone https://github.com/ta-oliver/infertrade.git
  ```
  - Alternative method: [clone](https://github.com/ta-oliver/infertrade/tree/main) source from website
- Change directory to infertrade
 
 ```
 cd infertrade
 ```
- Run setup file.
 ```
 python setup.py install
 ```
OR
 ```
 py setup.py install
 ```
- Installation should complete successfully without errors
