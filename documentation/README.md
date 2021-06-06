Ensure that requirements in requirements_dev are full filled.

Make changes to conf.py as per the requirements.

To initialize the Sphinx source directory execute the following inside documentation:

`sphinx-quickstart`

Currently the directory has been initialized. 

Use sphinx-apidoc to generate reStructuredText files from source code by executing following command:

`sphinx-apidoc -f -o <path-to-output> <path-to-module>`

Add the generated .rst file into index.rst toctree

Finally to build document you can execute following command:

`make latexpdf`

Additionally you can also build HTML by executing following command:

`make html`