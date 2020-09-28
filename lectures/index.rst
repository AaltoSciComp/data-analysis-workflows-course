=========================================
Data analysis workflows with R and Python
=========================================

.. admonition:: Attending the course 5.-14.10.?
                                                                    
   `See the course page here
   <https://scicomp.aalto.fi/training/scip/data-analysis/>`__,
   below is the course material.

Data analysis is nowadays at the center of almost all scientific fields.
Whether a researcher is doing experiments, running simulations or analyzing
datasets, at some point of their carreer they will be required to do data
analysis.

R and Python are two languages that have a rich and powerful data
analysis libraries and many researchers use them to build their data analysis
workflows. However, these libraries have been designed to work optimally
in certain types of workflows. Thus if one wants to create reproducible,
scalable and efficient data analysis workflows it is important to understand
how to design the workflow from the get-go.

This course contains four chapters:

    - First chapter is about understanding how data analysis workflows
      are commonly designed and how one should go about designing a new
      data analysis pipeline.
    - Second chapter is about data ingestion, tidy data format, and efficient
      data formats for input and output.
    - Third chapter is about calculating statistics, doing modeling and
      utilizing data analysis libraries.
    - Fourth chapter is about how one should think about scaling and how
      it can achieved.

Throughout this course the material has an underlying theme of three
principles that one should keep in mind throughout the course:

    - Recognizing a pattern - Different languages, libraries, workflows etc.
      have common features. We will try to find these patterns so that we
      familiarize ourselves with them.
    - Replicating a pattern - After we have recognized some patterns it is a good
      idea to copy said pattern and see if it works as we assumed.
    - Generalizing a pattern - Once we have a grasp of a how our pattern works
      we can think about how we could use it in a different context. That is,
      how we can generalize it.


.. prereq::

   - Knowledge of Python or R in the context of scientific computing.
     If you're unsure, our
     `Python for Scientific computing-course <https://aaltoscicomp.github.io/python-for-scicomp>`_
     is a good starting point.

   - Software installed via conda as described in :doc:`installation instructions <installation>`.

.. csv-table::
   :widths: auto
   :delim: ;

   120 min ; :doc:`chapter-1-understanding`
   120 min ; :doc:`chapter-2-data`
   120 min ; :doc:`chapter-3-modeling`
   120 min ; :doc:`chapter-4-scaling`

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Course lessons

   chapter-1-understanding
   chapter-2-data
   chapter-3-modeling
   chapter-4-scaling

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: References

   installation

.. _learner-personas:

Who is the course for?
----------------------

Researchers who are using or will soon be using R and Python
for data analysis, who know how to program with these languages, but do not
necessarily know what are the best practices for data analysis.

The course material is available in both R and Python, but this
is not a course on the basics of scientific programming. If you wish to
prep up your scientific programming skills, we recommend taking our
`Python for Scientific Computing <https://aaltoscicomp.github.io/python-for-scicomp/>`_-course.

See also
--------





Credits
-------

