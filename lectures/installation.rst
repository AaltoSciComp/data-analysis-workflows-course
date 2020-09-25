=====================
Software installation
=====================

******************
Conda installation
******************

This course will use `Conda <https://docs.continuum.io/anaconda/install/>`_
as the base installation platform as it can provide both Python and R
installations for multiple operating systems.

We recommend using
`Anaconda <https://www.anaconda.com/products/individual>`_,
but Miniconda will work as well.

*****************************
Installing course environment
*****************************

All of the required software is included in the course's
`environment.yml <https://raw.githubusercontent.com/AaltoSciComp/data-analysis-workflows-course/master/environment.yml>`_:

.. literalinclude:: ../environment.yml

To install the environment, use the following instructions based on your operating system.

Linux
=====

In terminal where the Anaconda installation is activated, clone the course repository with::

  git clone https://github.com/AaltoSciComp/data-analysis-workflows-course.git
  cd data-analysis-course

After this, in the course repository, run::

  conda env create environment.yml

If you wish to change the environment name from the default (`dataanalysis`), use::

  conda env create -n env_name environment.yml

Now you can launch a jupyterlab instance for running the exercises::

  jupyter-lab

Windows
=======

Start Anaconda Navigator. From the navigator, start **CMD.exe Prompt**. In the
prompt, install git with::

  conda install git

After installation, clone the course repository with::

  git clone https://github.com/AaltoSciComp/data-analysis-workflows-course.git
  cd data-analysis-course

After this, in the course repository, run::

  conda env create environment.yml

If you wish to change the environment name from the default (`dataanalysis`),
use::

  conda env create -n env_name environment.yml

You can now close the terminal. In the Anaconda Navigator **Home**-tab, next to
**Applications on**, switch from `base (root)` environment to `dataanalysis`.
Now you can launch a jupyterlab instance by clicking on **JupyterLab**.

If your jupyterlab instance says something like "X needs to be added to
the build", just click **Build**.

