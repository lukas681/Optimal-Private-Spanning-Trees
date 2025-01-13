=======================
Private Gradual Release
=======================


Experiments with xxx
The main experiments are located at src/Notebooks.

---------------
Getting Started
---------------

The followig guide sets up the environment used for the experiments.
All commands have been tested under ``WIN64`` and ``LNX`` as well.
Alternatively, use the corresponding ``conda`` commands.

* [MacOS] For MacOS set up the enviroment with::

        brew install virtualenv # only if not there
        virtualenv ~/.virtualenvs/private-gradual-releases --python=python3.9
        source ~/.virtualenvs/private-gradual-releases/bin/activate

* [WNX64] For Windows use the following (replae with your python version)::

        python3.9 -m venv private-gradual-releases
        private-gradual-releases/Scripts/activate

Install the dependencies from the ``requirements.txt``::

    pip install -r requirements.txt

===============
Troubleshooting
===============

If jupyter does not install, make sure to have pip in the latest version and try to install juypter seperately::

    ~/.virtualenvs/private-gradual-releases/bin/python -m ensurepip
    pip install jupyter

In case of ``python > 3.10``, run first::

    pip install setuptools

and then::

    ~/.virtualenvs/priv-mst/bin/python -m pip install pip --upgrade

Also feel free to use the environment in the IDE of your choice. (might give some problems on WIN64 systems).
If you are using DataSpell, make sure to add the correct interpreter in the setting.

-------
Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.