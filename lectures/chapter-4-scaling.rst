==================
Chapter 4: Scaling
==================

*******************************************************
Different resources involved in data analysis pipelines
*******************************************************

Modern data analysis pipelines can be very expensive computationally. In order
to scale these pipelines up, we'll need to first understand the resources that
these pipelines require. These resources are:

1. Processors (CPUs)
2. Memory (RAM)
3. Storage space
4. Time

Let's look at them one by one.

Processors as a resource
========================

Processor is the quintessential resource when it comes to data analysis. It
is used throughout the pipeline from data loading to data analysis and thus
it is important to know some features about them.

Firstly, modern processors are built from multiple cores. Sometimes these
cores can house multiple threads. This is called hyperthreading.

.. image:: images/processor.svg
    :align: center

Secondly, operations on data are done as
`instructions <https://en.wikipedia.org/wiki/Instruction_set_architecture>`_
inside the cores. These instructions handle calculations such as addition,
multiplication etc.. However, in order to get maximum throughput of finished
instructions, all modern CPU architectures have multiple layers of data caching
and prefetching that try to keep the calculating parts of the CPU as busy as
possible.

Data is read to the cache in blocks of data called cache lines. If required data
is not found in the cache, the data needs to be loaded from the system RAM,
which results in a significant performance penalty. This is called a
`cache miss <https://en.wikipedia.org/wiki/CPU_cache#Cache_miss>`_.

.. image:: images/processor-cache.svg
    :align: center

This caching procedure can be helped by keeping the data in memory as a
contiguous array. Both R vectors and numpy ndarrays are contiguous. They have
so-called
`row-major-ordering <https://en.wikipedia.org/wiki/Row-_and_column-major_order>`_.
It is also important to keep this order in mind when doing operations with
multidimensional arrays.

Another important feature of modern processors is that they support vectorized
instructions (AVX, AVX2, AVX512). These dramatically improve the performance
when one does the same operation for multiple pieces of data e.g. elementwise
addition. R, numpy and mathematical libraries that they use such as MKL, BLAS
LAPACK, FFTW etc. use these operations straight out of the box, if the program
is written to use functions from these packages.

We can test the effect of vectorization by looking at the following example
that adds to a zero array.

.. tabs::

  .. tab:: Python

    .. code-block:: python

        n_zeros = 10000
        times = 1000

        z = np.zeros(n_zeros)

        time_for_1 = time.time()
        for t in range(times):
            for i in range(n_zeros):
                z[i] = z[i] + 1
        time_for_2 = time.time()

        time_for = time_for_2-time_for_1

        z = np.zeros(n_zeros)

        time_vec_1 = time.time()
        for times in range(times):
            z = z + 1
        time_vec_2 = time.time()

        time_vec = time_vec_2-time_vec_1

        print("""
        Time taken:

        For loop: %.2g
        Vectorized operation: %.2g

        Speedup: %.0f
        """ % (time_for, time_vec, time_for/time_vec))
        

        Time taken:

        For loop: 4.5
        Vectorized operation: 0.0056

        Speedup: 801



  .. tab:: R

    .. code-block:: R

        n_zeros <- 10000
        ntimes <- 1000

        z <- numeric(n_zeros)

        time_for_1 <- Sys.time()
        for (t in seq(ntimes)) {
            for (i in seq(1,n_zeros)) {
                z[i] <- z[i] + 1
            }
        }
        time_for_2 <- Sys.time()

        time_for <- time_for_2 - time_for_1

        z <- numeric(n_zeros)

        time_vec_1 <- Sys.time()
        for (t in seq(ntimes)) {
            z <- z + 1
        }
        time_vec_2 <- Sys.time()

        time_vec <- time_vec_2 - time_vec_1

        cat(sprintf("Time taken:\n\nFor loop: %.2g\nVectorized operation: %.2g\n\nSpeedup: %.2f", time_for, time_vec, time_for/as.double(time_vec, unit='secs')))

        
        Time taken:

        For loop: 0.61
        Vectorized operation: 0.018

        Speedup: 33.61


Time as a resource
==================

Time is naturally one of the resources 

.. tabs::

  .. tab:: Python

    .. code-block:: python

        pass

  .. tab:: R

    .. code-block:: R

        NULL