Ensure that requirements in `requirements-dev.txt` are fulfilled.

Make changes to conf.py as per the requirements.

To initialize the Sphinx source directory execute the following inside documentation:

`sphinx-quickstart`

Currently the directory has been initialized. 

Use `sphinx-apidoc` to generate reStructuredText files from source code by executing the following command:

`sphinx-apidoc -f -o <path-to-output> <path-to-module>`

Add the generated `.rst` file into the `index.rst` toctree.

Finally, to build documentation you can execute the following command:

`make latexpdf`

Additionally, you can build HTML by executing the following command:

`make html`
