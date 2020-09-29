===========================
Chapter 2: Data preparation
===========================

Tidy data - Data in memory
--------------------------

Both R (``data.frame`` and ``tibble``) and pandas (``DataFrame``) store the
data as columnar vectors. They are in a so-called column-oriented format.

.. image:: images/dataframe.svg

When the data is organized in a way where each column corresponds to a
variable and each row corresponds to a observation, we consider the dataset
**tidy**.

This image from Hadley Wickham's excellent book
`R for Data Science <https://r4ds.had.co.nz/>`_ visualizes the idea:

.. image:: https://raw.githubusercontent.com/hadley/r4ds/master/images/tidy-1.png

The name "tidy data" comes from Wickham's
`paper (2014) <https://vita.had.co.nz/papers/tidy-data.pdf>`_, that describes
the ideas in great detail.

The choice of this data format is by no means a one man show: in fact Wickham
cites Wes McKinney's
`paper (2010) <http://conference.scipy.org/proceedings/scipy2010/pdfs/mckinney.pdf>`_
on Pandas' chosen data format as an example of a tidy data format. In fact both
`R <https://rstudio.com/wp-content/uploads/2015/02/data-wrangling-cheatsheet.pdf>`_
and `Pandas <https://pandas.pydata.org/Pandas_Cheat_Sheet.pdf>`_ developers
recommend using tidy data.

Main reasons why this data format is recommended are:

1. It is easy to understand and helps to internalize how your data is
   organized.
2. In tidy data format all observations in a variable have the same data
   type. This means that columns can be stored as vectors. This saves memory
   and allows for vectorized calculations that are much faster.
3. It is easy for developers to create methods as they can assume how the data
   is organized.

There are cases where tidy data might not be the optimal format. For example, if
your problem has matrices, you would not want to store it as rows and columns,
but as a two-dimensional array. Another situation might arise when your data is
in some other binary format e.g. image data. But in these cases you might
want to use tidy data to store e.g. image names or model parameters.

In this course our data is mostly in tidy format and if it's not in that
format, we'll want to convert our raw data into it as soon as possible.