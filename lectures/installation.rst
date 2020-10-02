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

To install the environment, use the following instructions based on
your operating system.  To learn more about environments in general,
see `the conda environment docs
<https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html>`__
(but there is more here than you need to know), if you can do the
steps below that is enough.

Linux and Mac OSX
=================

In terminal where the Anaconda installation is activated, clone the course repository with::

  git clone https://github.com/AaltoSciComp/data-analysis-workflows-course.git
  cd data-analysis-workflows-course

After this, in the course repository, run::

  conda env create environment.yml

If you wish to change the environment name from the default (``dataanalysis``), use::

  conda env create -n env_name environment.yml

Then activate the environment (if you didn't call it ``dataanalysis``,
replace it with the name you used)::

  conda activate dataanalysis

Now you can launch a jupyterlab instance for running the exercises::

  jupyter-lab

Windows
=======

Download the environment file
`environment.yml <https://raw.githubusercontent.com/AaltoSciComp/data-analysis-workflows-course/master/environment.yml>`_
somewhere.

Start Anaconda Navigator. From the navigator, go to **Environments**.

From bottom, click **Import**. Choose **Name** - *dataanalysis* and
for **Specification File** choose the downloaded environment file.

The environment creation process can take a long time, as the
environment is quite big.

After installation, in the Anaconda Navigator **Home**-tab, next to
**Applications on**, switch from ``base (root)`` environment to ``dataanalysis``.
Now you can launch a jupyterlab instance by clicking on **JupyterLab**.

If your jupyterlab instance says something like "X needs to be added to
the build", just click **Build** and continue forward. If the build gives
an error later on, just ignore it.

Now press the **+**-button on the top left and under **Console**, choose
**Python 3**.

In the console at the botton, type the following:

.. code-block:: python

    import git

    git.Git().clone('https://github.com/AaltoSciComp/data-analysis-workflows-course.git')

This will download the course repository to folder ``data-analysis-workflows-course``.
You can now go the folder and commence the testing.


Testing your installation
=========================

This workshop requires that you are familiar with Jupyter notebooks and how to run
them. In the git repository that you have downloaded for the installation, you will
find a notebook called ``download_datasets.ipynb``. Open it and run it. This will
download multiple datasets into the subfolder `data/`. You can try loading some of
these datasets to make sure the download went through. Next, open the notebook for
the first exercise, you will find it under ``X_exercises/ch1-X-ex1.ipynb``
(replace ``X`` with ``python`` or ``r``). Make sure you are able to fully run the
notebook. In case of installation issues, join us in the pre-workshop meeting
(you have received details via email).



