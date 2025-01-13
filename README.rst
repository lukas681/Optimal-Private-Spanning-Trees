==================
Optimal Private Minimum Spanning Trees using Output Perturbation
==================


Experiments with Privately Releasing Minimum Spanning Trees
The main experiments are located at src/Notebooks

Getting Started
--------

Make sure python 3 virtual environments are install::

    brew install virtualenv

We use the following environment to run the experiments.
Requirements are stored inside requirements.txt:



.. code-block:: bash

    virtualenv ~/.virtualenvs/priv-mst --python=python3.9
    source ~/.virtualenvs/priv-mst/bin/activate
    pip install -r requirememts.txt


If jupyter does not install, make sure to have pip in the latest version, install juypter seperately::

    ~/.virtualenvs/priv-mst/bin/python -m ensurepip
    pip install jupyter

For python > 3.10, run first::

    pip install setuptools

and then

    ~/.virtualenvs/priv-mst/bin/python -m pip install pip --upgrade

Also feel free to use the environment in the IDE of your choice. (might give some problems on WIN64 systems).
If you are using DataSpell, make sure to add the correct interpreter in the setting.

Tested on WIN64 and MACOS

Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.