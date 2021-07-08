![](images/infertrade_logo.png)

# Welcome!

**New to Anaconda environment , Not a problem. It is a great start. Let's look in berief about about Anaconda Software.**

### What made us in Infertrade to make a documentation on this ?
* If you are dealing with data_science, Machine Learing then you might be found of Anaconda software. It is awesome as it provides flexible environment to make your job easy and fast.

* Let’s say, you have already multiple versions of python installed and you want to use python 3.7x for our product but Currently you are using python 3.9x in your project then you will find some difficulties in making virtual environment with python 3.7x using virtualenv , venv or any other any other tool except conda. In case, 
somehow you will manage to install without using conda but there might be again a problem while installing some of the dependencies. So, how to deal with this kind of 
situation?

   _The answer is using conda to create virtual environment that help us to install required python version for us._ 

* Once you are familiar with Anaconda Software and conda package manager we will make virtual environment required for our software.


**Conda is a package manager for Anaconda prompt. Anaconda propmt and Conda is installed along with anaconda software i.e. they dont need seperate installation.**

* Anaconda is a distribution of the Python and R programming languages for scientific computing, that aims to simplify package management and deployment.

* The distribution includes data-science packages suitable for Windows, Linux, and macOS.

* Anaconda prompt is like command prompt through which we can install and manage data science libraries using conda pacakge manager.

*You can install anaconda for your system from the link below:*
[Anaconda Installation](https://www.anaconda.com/products/individual "Install anaconda")

AFTER INSTALLING ANACONDA

*Below is the link to conda guide for dealing with conda virtual environment.*

[conda-cheatSheet](https://docs.conda.io/projects/conda/en/4.6.0/_downloads/52a95608c49671267e40c689e0bc00ca/conda-cheatsheet.pdf "conda-cheatsheet")

**Let’s create a virtual environment having python 3.7x using conda with virtual environment name – infertradenv. You are always free to give the name of your choice.**

* Step1: Open Anaconda prompt and make your way to desire drive (probably C drive).
*  Step2: make a sub folder “infertrade” in user’s folder then cd infertrade.
* Step3: conda create --name infertradenv python = 3.7
* Step4: conda env list
* Step5: conda activate infertradenv
* Step6: python --version

   *  make sure you have python 3.7x.
* Step7: curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py

* Step8: py get-pip.py

* Step9: pip install infertrade

* Step10: pip install seaborn

* Step11: pip install jupyter notebook

* Step12: Open jupyter notebook in activated virtual environment by running the command in anaconda prompt  (jupyter notebook)

**Congratulations! You are all set to work with virtual environment with the infertrade package.**

Below, we have shown how the above steps in more specified way. Run the code one by one.

```Anaconda promt
Open anaconda propmt
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

jupter notebook

```

**From opened jupyter notebook run our example.ipynb file to experince infertrade package.**

If you are new to jupyter notebook you can check the link below:
[Jupyter](https://jupyter.org "jupyter")
