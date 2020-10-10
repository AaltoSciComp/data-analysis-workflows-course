===================
Chapter 3: Modeling
===================

*****************
Types of modeling
*****************

In this chapter we will focus on doing modeling on our datasets. In our case
the term "modeling" refers to any activity that takes in our data and produces
a result. We'll be doing four differing types of modeling:

1. Creating summaries of our data by calculating statistics
   (e.g. mean, variance).
2. Using bootstrapping methods to calculate statistical moments.
3. Fitting a function into our data.
4. Running a machine learning model with training and test steps.

To accomplish these task, we'll be using various features of our frameworks:

1. Groupings - Grouping data by a common variable for efficient summaries and
   plotting.
2. Apply- and map-functions - Functions that run a function for all rows of
   our data structure.
3. Nesting/stacking - Storing data frames in our data frames.

**********************************
Grouping data by a common variable
**********************************

In grouping one determines one or more variables to use as grouping index and
then these indices are used to apply some function to each subgroup. They are
especially useful when one is calculating some statistics from each subgroup,
doing multiple plots or determining plot colors. 

Let's consider the dataset presented below. It is contains the number of files
stored on Aalto University's Triton cluster's file system. Columns are:

- File size rounded down to the nearest exponent of 2 (e.g. 3 MB file would
  be counted to row with size 2097152 (2^21 = 2097152))
- File modification date in months before 31.12.2020 (e.g. file written in
  3.9.2020 would be 3)
- Number of files

.. tabs::

  .. tab:: Python

    .. code-block:: python
    
        def load_filesizes(filesizes_file):
            filesizes = pd.read_table(filesizes_file, sep='\s+', names=['Bytes','MonthsTo2021', 'Files'])

            # Remove empty files
            filesizes = filesizes[filesizes.loc[:,'Bytes'] != 0]
            # Create a column for log2 of bytes
            filesizes['BytesLog2'] = np.log2(filesizes.loc[:, 'Bytes'])
            filesizes.loc[:,'BytesLog2'] = filesizes.loc[:,'BytesLog2'].astype(np.int64)
            # Determine total space S used by N files of size X during date D: S=N*X 
            filesizes['SpaceUsage'] = filesizes.loc[:,'Bytes']*filesizes.loc[:,'Files']
            # Determine file year and month from the MonthsTo2021-column
            filesizes['TotalMonths'] = 2021*12 - filesizes['MonthsTo2021'] - 1
            filesizes['Year'] = filesizes['TotalMonths'] // 12
            filesizes['Month'] = filesizes['TotalMonths'] % 12 + 1
            filesizes['Day'] = 1

            # Set year for really old files and files with incorrect timestamps
            invalid_years = (filesizes['Year'] < 2010) | (filesizes['Year'] > 2020)
            filesizes.loc[invalid_years, ['Year','Month']] = np.NaN

            # Create Date and get the name for the month
            filesizes['Date'] = pd.to_datetime(filesizes[['Year', 'Month', 'Day']])
            filesizes['MonthName'] = filesizes['Date'].dt.month_name()
            # Sort data based on Date and BytesLog2
            filesizes.sort_values(['Date','BytesLog2'], inplace=True)
            # Remove old columns
            filesizes.drop(['MonthsTo2021','TotalMonths', 'Day'], axis=1, inplace=True)
            return filesizes

        filesizes = load_filesizes('../data/filesizes_timestamps.txt')
        filesizes.head()

  .. tab:: R

    .. code-block:: R

        load_filesizes <- function(filesizes_file){
            filesizes <- read_table2(filesizes_file, col_names=c('Bytes','MonthsTo2021', 'Files'))

            filesizes <- filesizes %>%
            # Remove empty files
                filter(Bytes != 0) %>%
                # Create a column for log2 of bytes
                mutate(BytesLog2 = log2(Bytes)) %>%
                # Determine total space S used by N files of size X during date D: S=N*X 
                mutate(SpaceUsage = Bytes*Files) %>%
                # Determine file year and month from the MonthsTo2021-column
                mutate(
                    TotalMonths = 2021*12 - MonthsTo2021 - 1,
                    Year = TotalMonths %/% 12,
                    Month = TotalMonths %% 12 +1,
                    Day = 1
                )

             # Set year for really old files and files with incorrect timestamps
            invalid_years = c((filesizes['Year'] < 2010) | (filesizes['Year'] > 2020))
            filesizes[invalid_years, c('Year','Month')] <- NaN

            # Create Date and get the name for the month
            filesizes <- filesizes %>%
                mutate(
                    Date = make_datetime(Year, Month, Day),
                    MonthName= month(Month, label=TRUE, locale='C'))
            filesizes <- filesizes %>%
                # Sort data based on Date and BytesLog2
                arrange(Date, BytesLog2) %>%
                # Remove old columns
                select(-MonthsTo2021,-TotalMonths,-Day)
            return(filesizes)
        }

        filesizes <- load_filesizes('../data/filesizes_timestamps.txt')
        head(filesizes)

Our parsed file contains columns for date, year, month, month name, the size of
files in two different formats, the number of files and the total space used by
the files. Let's say we're interested in the how the number of file has
increased each year. To do this, we'll first limit our focus on the relevant
columns.

.. tabs::

  .. tab:: Python

    .. code-block:: python
    
        # Drop rows with NaNs (invalid years)
        newfiles_relevant = filesizes.dropna(axis=0)
        # Pick relevant columns
        newfiles_relevant = newfiles_relevant.loc[:,['Year','BytesLog2','Files']]
        # Change year to category for prettier plotting
        newfiles_relevant['Year'] = newfiles_relevant['Year'].astype('int').astype('category')
        newfiles_relevant.head()

  .. tab:: R

    .. code-block:: R
    
        newfiles_relevant <- filesizes %>%
            # Drop rows with NaNs (invalid years)
            drop_na() %>%
            # Pick relevant columns
            select(Year, Files) %>%
            # Change year to category for prettier plotting
            mutate(Year=as.factor(Year))
            head(newfiles_relevant)


Now, we'll want to group our data based on the year-column (``Year``) and
calculate the total number of files (``Files``) across all rows (all dates
and files sizes).

.. tabs::

  .. tab:: Python

    .. code-block:: python
    
        print(newfiles_relevant.shape)
        
        newfiles_yearly_sum = newfiles_relevant.groupby(['Year']).agg('sum')
        
        print(newfiles_yearly_sum.shape)
        newfiles_yearly_sum.head()

  .. tab:: R

    .. code-block:: R

        glimpse(newfiles_relevant)

        newfiles_yearly_sum <- newfiles_relevant %>%
            group_by(Year) %>%
            summarize(Files=sum(Files))

        glimpse(newfiles_yearly_sum)
        head(newfiles_yearly_sum)
        

.. tabs::

  .. tab:: Python

    .. code-block:: python
    
        pass

  .. tab:: R

    .. code-block:: R

        NULL

In Python we see that the output of 
`agg <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.agg.html>
is still grouped and for plotting, we'll want
to reset the grouping. R
`summarise <https://dplyr.tidyverse.org/reference/summarise.html>`
removes the last layer of groupings, but let's
verify that the data is ungrouped. 

.. tabs::

  .. tab:: Python

    .. code-block:: python
    
        filesizes_yearly_sum = filesizes_yearly_sum.reset_index()

  .. tab:: R

    .. code-block:: R

        NULL


Let's plot this data in a bar plot:

.. tabs::

  .. tab:: Python

    .. code-block:: python
    
        sb.barplot(x='Year', y='Files', data=filesizes_yearly_sum, ci=None)

  .. tab:: R

    .. code-block:: R

        NULL

.. tabs::

  .. tab:: Python

    .. code-block:: python
    
        filesizes_yearly_sum = filesizes_yearly_sum.reset_index()

  .. tab:: R

    .. code-block:: R

        NULL
