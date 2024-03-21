# Getting started

## Installation
FM2PROF is a package written in Python. If you're familiar with Python, you can install FM2PROF with [`pip`][pip], the Python package manager. If not, we recommend the [executable]. 


[pip]: #with-pip
[executable]: #executable

### With pip

FM2PROF is published as a [Python package] and can be installed with `pip`, ideally by using a [virtual environment] and using Python 3.10 or higher. Open up a terminal and install FM2PROF with:

=== "Latest"

    ``` bash
    pip install fm2prof
    ```

=== "Specific version"

    ``` bash
    pip install fm2prof="2.2.8"
    ```
=== "update existing install"

    ``` bash
    pip install fm2prof --upgrade
    ```


This will also install compatible versions of all dependencies like `numpy`, and `pandas`. 

[Python package]: https://pypi.org/project/fm2prof
[virtual environment]: https://realpython.com/what-is-pip/#using-pip-in-a-python-virtual-environment

### Executable

FM2PROF is compiled to an executable for Windows (version 1.5.4 and higher). Download them from [Releases](https://github.com/Deltares/Fm2Prof) on the Github home page.

The executable provides a command line interface (CLI). To use it, make
sure it is available in your Path. To view available options, open a
Terminal and type:

``` shell
FM2PROF --help
```

!!! warning
    The first time you call the executable, it may take a long time to
    produce results. This is because your OS needs to unpack the executable.


