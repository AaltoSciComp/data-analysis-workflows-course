===========================
Chapter 2: Data preparation
===========================

.. important::

    Commands run on this chapter are present in the
    ``X_exercises/ch2-X-lectures.ipynb``, where ``X`` is the programming
    language.

*********
Tidy data
*********

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

**********************
Simple data operations
**********************

Loading data from CSVs
======================

Let's start with loading data in the most common data format: csv. Quite often
datasets are provided in this format because it is human readable and easy to
transfer among systems.

Let's consider a ``atp_players.csv``-dataset that contains ATP ranks of men's
singles tennis players.

Let's load the data with the ``read_csv``-function:

.. tabs::

  .. tab:: Python
  
    `pandas.read_csv <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html>`_

    .. code-block:: python
    
        atp_players = pd.read_csv('../data/atp_players.csv', names=['player_id', 'first_name', 'last_name', 'hand', 'birth_date', 'country_code'])
        atp_players.head()
        	player_id 	first_name 	last_name 	hand 	birth_date 	country_code
        0 	100001 	Gardnar 	Mulloy 	R 	19131122.0 	USA
        1 	100002 	Pancho 	Segura 	R 	19210620.0 	ECU
        2 	100003 	Frank 	Sedgman 	R 	19271002.0 	AUS
        3 	100004 	Giuseppe 	Merlo 	R 	19271011.0 	ITA
        4 	100005 	Richard Pancho 	Gonzales 	R 	19280509.0 	USA

  .. tab:: R
  
    `Tidyverse read_csv <https://readr.tidyverse.org/reference/read_delim.html>`_

    .. code-block:: R
    
        atp_players <- read_csv('../data/atp_players.csv', col_names=c('player_id', 'first_name', 'last_name', 'hand', 'birth_date', 'country_code'))
        Parsed with column specification:
        cols(
          player_id = col_double(),
          first_name = col_character(),
          last_name = col_character(),
          hand = col_character(),
          birth_date = col_double(),
          country_code = col_character()
        )
        player_id	first_name	last_name	hand	birth_date	country_code
        100001 	Gardnar 	Mulloy 	R 	19131122 	USA
        100002 	Pancho 	Segura 	R 	19210620 	ECU
        100003 	Frank 	Sedgman 	R 	19271002 	AUS
        100004 	Giuseppe 	Merlo 	R 	19271011 	ITA
        100005 	Richard Pancho	Gonzales 	R 	19280509 	USA
        100006 	Grant 	Golden 	R 	19290821 	USA

This function not only parses the text, but also tries to convert the columns
to a best possible fata types. To check column data types, use:

.. tabs::

  .. tab:: Python

    .. code-block:: python
    
        print(iris.dtypes)
        player_id         int64
        first_name       object
        last_name        object
        hand             object
        birth_date      float64
        country_code     object
        dtype: object

  .. tab:: R

    .. code-block:: R
    
        str(atp_players)
    
        Classes ‘spec_tbl_df’, ‘tbl_df’, ‘tbl’ and 'data.frame':	54938 obs. of  6 variables:
         $ player_id   : num  1e+05 1e+05 1e+05 1e+05 1e+05 ...
         $ first_name  : chr  "Gardnar" "Pancho" "Frank" "Giuseppe" ...
         $ last_name   : chr  "Mulloy" "Segura" "Sedgman" "Merlo" ...
         $ hand        : chr  "R" "R" "R" "R" ...
         $ birth_date  : num  19131122 19210620 19271002 19271011 19280509 ...
         $ country_code: chr  "USA" "ECU" "AUS" "ITA" ...
         - attr(*, "spec")=
          .. cols(
          ..   player_id = col_double(),
          ..   first_name = col_character(),
          ..   last_name = col_character(),
          ..   hand = col_character(),
          ..   birth_date = col_double(),
          ..   country_code = col_character()
          .. )

The ``head``-function can be used to show the first few rows of our dataset.

.. tabs::

  .. tab:: Python

    .. code-block:: python
    
        atp_players.head()
        
        	player_id 	first_name 	last_name 	hand 	birth_date 	country_code
        0 	100001 	Gardnar 	Mulloy 	R 	19131122.0 	USA
        1 	100002 	Pancho 	Segura 	R 	19210620.0 	ECU
        2 	100003 	Frank 	Sedgman 	R 	19271002.0 	AUS
        3 	100004 	Giuseppe 	Merlo 	R 	19271011.0 	ITA
        4 	100005 	Richard Pancho 	Gonzales 	R 	19280509.0 	USA

  .. tab:: R

    .. code-block:: R
    
        head(atp_players)
    
        player_id	first_name	last_name	hand	birth_date	country_code
        100001 	Gardnar 	Mulloy 	R 	19131122 	USA
        100002 	Pancho 	Segura 	R 	19210620 	ECU
        100003 	Frank 	Sedgman 	R 	19271002 	AUS
        100004 	Giuseppe 	Merlo 	R 	19271011 	ITA
        100005 	Richard Pancho	Gonzales 	R 	19280509 	USA
        100006 	Grant 	Golden 	R 	19290821 	USA 



Creating and removing new columns
=================================

Let's start by converting the birth date column into an actual time stamp.

.. tabs::

  .. tab:: Python
  
    `pandas.to_datetime <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.to_datetime.html>`_

    .. code-block:: python
    
        atp_players['birth_date'] = pd.to_datetime(atp_players['birth_date'], format='%Y%m%d', errors='coerce')
        print(atp_players.dtypes)
        
        player_id                int64
        first_name              object
        last_name               object
        hand                    object
        birth_date      datetime64[ns]
        country_code            object
        dtype: object

  .. tab:: R
  
    `Tidyverse mutate <https://dplyr.tidyverse.org/reference/mutate.html>`_
    
    `Lubridate parse_date_time <https://lubridate.tidyverse.org/reference/parse_date_time.html>`_

    .. code-block:: R
    
        library(lubridate)
        atp_players <- atp_players %>%
            mutate(birth_date=parse_date_time(birth_date, order='%Y%m%d'))
        str(atp_players)

        Warning message:
        “ 125 failed to parse.”

        Classes ‘spec_tbl_df’, ‘tbl_df’, ‘tbl’ and 'data.frame':	54938 obs. of  6 variables:
         $ player_id   : num  1e+05 1e+05 1e+05 1e+05 1e+05 ...
         $ first_name  : chr  "Gardnar" "Pancho" "Frank" "Giuseppe" ...
         $ last_name   : chr  "Mulloy" "Segura" "Sedgman" "Merlo" ...
         $ hand        : chr  "R" "R" "R" "R" ...
         $ birth_date  : POSIXct, format: "1913-11-22" "1921-06-20" ...
         $ country_code: chr  "USA" "ECU" "AUS" "ITA" ...

In our current situation we have separate columns for first and last names.
Let's join these columns into one column called ``name``:

.. tabs::

  .. tab:: Python

    .. code-block:: python
    
        atp_players['name'] = atp_players['last_name'] + ', ' + atp_players['first_name']
        
        atp_players.head()
        
         	player_id 	first_name 	last_name 	hand 	birth_date 	country_code 	name
        0 	100001 	Gardnar 	Mulloy 	R 	19131122.0 	USA 	Mulloy, Gardnar
        1 	100002 	Pancho 	Segura 	R 	19210620.0 	ECU 	Segura, Pancho
        2 	100003 	Frank 	Sedgman 	R 	19271002.0 	AUS 	Sedgman, Frank
        3 	100004 	Giuseppe 	Merlo 	R 	19271011.0 	ITA 	Merlo, Giuseppe
        4 	100005 	Richard Pancho 	Gonzales 	R 	19280509.0 	USA 	Gonzales, Richard Pancho

  .. tab:: R
  
    `Tidyverse unite <https://tidyr.tidyverse.org/reference/unite.html>`_

    .. code-block:: R
    
        atp_players <- atp_players %>%
            unite(name, last_name, first_name, sep=', ', remove=FALSE)

        head(atp_players)

        player_id	name	first_name	last_name	hand	birth_date	country_code
        100001 	Mulloy, Gardnar 	Gardnar 	Mulloy 	R 	19131122 	USA
        100002 	Segura, Pancho 	Pancho 	Segura 	R 	19210620 	ECU
        100003 	Sedgman, Frank 	Frank 	Sedgman 	R 	19271002 	AUS
        100004 	Merlo, Giuseppe 	Giuseppe 	Merlo 	R 	19271011 	ITA
        100005 	Gonzales, Richard Pancho	Richard Pancho 	Gonzales 	R 	19280509 	USA
        100006 	Golden, Grant 	Grant 	Golden 	R 	19290821 	USA

Now we can drop our unneeded columns:

.. tabs::

  .. tab:: Python
  
    `pandas.DataFrame.drop <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.drop.html>`_

    .. code-block:: python
    
        atp_players.drop(['first_name','last_name'], axis=1, inplace=True)
        atp_players.dtypes
        
        player_id         int64
        hand             object
        birth_date      float64
        country_code     object
        name             object
        dtype: object

  .. tab:: R
  
    `Tidyverse select <https://dplyr.tidyverse.org/reference/select.html>`_

    .. code-block:: R
    
        atp_players <- atp_players %>%
            select(-first_name, -last_name)

        str(atp_players)
        
        Classes ‘tbl_df’, ‘tbl’ and 'data.frame':	54938 obs. of  5 variables:
         $ player_id   : num  1e+05 1e+05 1e+05 1e+05 1e+05 ...
         $ name        : chr  "Mulloy, Gardnar" "Segura, Pancho" "Sedgman, Frank" "Merlo, Giuseppe" ...
         $ hand        : chr  "R" "R" "R" "R" ...
         $ birth_date  : num  19131122 19210620 19271002 19271011 19280509 ...
         $ country_code: chr  "USA" "ECU" "AUS" "ITA" ...

Categorical data format
=======================

When working with string data that has well defined categories, it is usually a
good idea to convert the data into categorical (Python) / factor (R) format.
In this format all unique strings are given an integer value and the string
array is converted into an integer array with this mapping. The unique strings
are called "categories" or "levels" of the categorical/factor array. 

Main benefits of using categorical data are:

- Easy to combine categories together.
- Helps with grouping and plot labeling.
- Reduced memory consumption.

Disadvantages include:

- For string arrays with completely unique values (e.g. our ``name``-column),
  most of the benefits are lost.
- Some models may recognize categorical data as numeric data as the underlying
  format in memory is an integer array. Check documentation of your modeling
  function whether it works with categorical data.

.. tabs::

  .. tab:: Python
  
    `Pandas categorical <https://pandas.pydata.org/pandas-docs/stable/user_guide/categorical.html>`_
    
    `Pandas apply <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.apply.html>`_

    .. code-block:: python
    
        print(atp_players['hand'].nbytes)
        atp_players[['country_code', 'hand']] = atp_players[['country_code', 'hand']].apply(lambda x: x.astype('category'))
        print(atp_players['country_code'].nbytes)
        print(atp_players['hand'].cat.categories)
        atp_players.dtypes

        439504
        111556
        Index(['A', 'L', 'R', 'U'], dtype='object')

        player_id          int64
        hand            category
        birth_date       float64
        country_code    category
        name              object
        dtype: object

  .. tab:: R
  
    `R factor <https://www.rdocumentation.org/packages/base/versions/3.6.2/topics/factor>`_
    
    `Tidyverse mutate <https://dplyr.tidyverse.org/reference/mutate_all.html>`_

    .. code-block:: R
    
        object.size(atp_players[['hand']])
        atp_players <- atp_players %>%
            mutate_at(c('country_code', 'hand'), as.factor)
        object.size(atp_players[['hand']])
        print(levels(atp_players[['hand']]))
        str(atp_players)
        
        439776 bytes
        220440 bytes
        [1] "A" "L" "R" "U"
        Classes ‘tbl_df’, ‘tbl’ and 'data.frame':	54938 obs. of  5 variables:
         $ player_id   : num  1e+05 1e+05 1e+05 1e+05 1e+05 ...
         $ name        : chr  "Mulloy, Gardnar" "Segura, Pancho" "Sedgman, Frank" "Merlo, Giuseppe" ...
         $ hand        : Factor w/ 4 levels "A","L","R","U": 3 3 3 3 3 3 2 3 3 3 ...
         $ birth_date  : num  19131122 19210620 19271002 19271011 19280509 ...
         $ country_code: Factor w/ 210 levels "AFG","AHO","ALB",..: 200 62 13 97 200 200 160 58 88 43 ...

`read_excel <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_excel.html>`_

`read_excel <https://readxl.tidyverse.org/reference/read_excel.html>`_