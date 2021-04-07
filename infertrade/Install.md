**How to Install Infertrade Package from scratch for windows.**
 
- Install Python 3.7.x 
 
    [Python Download](https://www.python.org/)
 
- Add Path to Environment Variables

 ![alt-text](https://github.com/ta-oliver/infertrade/blob/holderfolyf-patch-1/docs/image/run_dialog_box.jpg)
 ![alt-text](https://github.com/ta-oliver/infertrade/blob/holderfolyf-patch-1/docs/image/2%20environment_variables.jpg)
 ![alt-text](https://github.com/ta-oliver/infertrade/blob/holderfolyf-patch-1/docs/image/3edit_environment_variables.jpg)
 ![alt-text](https://github.com/ta-oliver/infertrade/blob/holderfolyf-patch-1/docs/image/4add_path.jpg)
 
- Install PIP package manager 
 
  - Download file from shell/cmd:
  ```
  curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
  ```
  - Alternatively download from browser
  ```
  https://bootstrap.pypa.io/get-pip.py
  ```
  - Navigate to directory in cmd and run following command
  ```
  py get-pip.py
  ```
- Add Path to Environment Variables for pip package manager.
```
setx PATH “%PATH%;C:\Python37\Scripts”
```
 
- Install infertrade package with pip install command.
```
pip install infretrade
```
- Script should run without errors, it will install dependencies as well.
 
- Ensure all dependencies installed by running the following command.
```
pip list
```
- Open the infertrade package with your favorite code editor and run test files to confirm everything is running in order.
 
## Alternative Installation Method from source.
 
- Clone Repo
```
git clone https://github.com/ta-oliver/infertrade.git
```
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

