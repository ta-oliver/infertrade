![](images/infertrade_logo.png)



# Welcome!

**New to Anaconda?**
**No problem, let's take a brief look into it**

### Why would the team at Infertrade make documentation regarding this?
* If you work with Data or Machine Learning then you might be fond of Anaconda software. It is great as it provides a flexible environment which makes your job faster and easier.

* Let’s say, you already have multiple versions of python installed and you want to use Python 3.7x for our product but currently, you are using python 3.9x in your project. In that case. You will experience some difficulties in creating a virtual environment with Python 3.7x using virtualenv, venv or any other tool except Conda. 
* In case you somehow manage to install it without using Conda you might still experience problems while installing some of the dependencies.
* So, how can deal with this kind of problem?

   -You can get around this issue by using Conda to create a virtual environment that will help install the required Python version.-

* Once you are familiar with Anaconda Software and the Conda package manager we will make a virtual environment required for our software.


**Conda is a package manager for Anaconda prompt which is installed alongside Conda and Anaconda Software meaning that they don't need to be installed separately.**

* Anaconda is a distribution of the Python and R programming languages for scientific computing, that aims to simplify package management and deployment.

* The distribution includes data science packages suitable for Windows, Linux, and macOS.

* Anaconda prompt is a terminal similar to a command prompt through which we can install and manage data science libraries using the Conda package manager.

*You can download and install Anaconda from the link below:*
[Anaconda Download](https://www.anaconda.com/products/individual "Install anaconda")

AFTER INSTALLING ANACONDA

*Below is the link to a Conda guide used for dealing with the virtual environment provided by Conda.*

[conda-cheatSheet](https://docs.conda.io/projects/conda/en/4.6.0/_downloads/52a95608c49671267e40c689e0bc00ca/conda-cheatsheet.pdf "conda-cheatsheet")

**Let’s create a virtual environment with python 3.7x using Conda, we went with the name – "infertradenv", but are always free to set the name of your choice.**

* Step 1: Open Anaconda prompt and navigate to your desired drive.

* Step 2: Make a sub folder called “infertrade” in your user’s folder, and then use the cd infertrade command.

# Follow the commands used in the following steps

* Step 3: conda create --name infertradenv python = 3.7

* Step 4: conda env list

* Step 5: conda activate infertradenv

* Step 6: python --version - This is to make sure you are using python 3.7

* Step 7: curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py

* Step 8: py get-pip.py

* Step 9: pip install infertrade

* Step 10: pip install seaborn

* Step 11: pip install Jupyter notebook

* Step 12: Open Jupyter notebook in you're activated virtual environment by running the command in anaconda prompt  (Jupyter notebook)

**Congratulations! You are all set to use your virtual environment with Infertrade.**

The following text is a representation of the installation process within Anaconda prompt.

```Anaconda prompt
Open anaconda prompt
c:users\<user_name>

#codes are listed below:

mkdir infertrade

cd infertrade

conda create --name infertradenv python=3.7

conda env list

conda activate infertradenv

curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py

py get-pip.py

pip install infertrade

pip install seaborn

pip install jupyter notebook

jupyter notebook

```

**After opening Jupyter notebook run our "example.ipynb" file to use Infertrade.**

In case you are new to Jupyter notebook the link below will take you to their website where you can find more documentation on how to use it.

[Jupyter](https://jupyter.org "jupyter")
