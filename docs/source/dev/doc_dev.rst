=======================
Build the documentation
=======================

.. contents:: Table of Contents

Install requirements
--------------------

Building the :stable_gym:`Stable Gym <>` documentation requires `sphinx`_,
the ``stable_gym`` python package and several plugins. All of the above can be
installed using the following `pip`_ command inside the ``./stable_gym`` folder:

.. code-block:: bash

    pip install -e .[docs]

.. _sphinx: http://www.sphinx-doc.org/en/master
.. _pip: https://pypi.org/project/pip/

Build the documentation
-----------------------

To build the `HTML`_ documentation, go into the :stable_gym:`docs/ <tree/main/stable_gym/docs>` directory and run the
``make html`` command. This command will generate the html documentation inside the ``docs/build/html`` directory. If the documentation is successfully built, you can also use the ``make linkcheck`` command to check for broken links.

.. note::
    Make sure you are in the Conda environment in which you installed the stable_gym package
    with it's dependencies.

.. _HTML: https://www.w3schools.com/html/

Deploying
---------

To deploy documentation to the Github Pages site for the repository, push the
documentation to the :stable_gym:`main <tree/main>` branch and run the ``make gh-pages`` command
inside the :stable_gym:`docs/ <tree/main/stable_gym/docs>` directory.
