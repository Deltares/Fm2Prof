# Contributing

This is brief guide on how to contribute to the code. This guide is 
written for Deltares developers. At the moment, we are not accepting 
external pull requests. 

## Set up development environment

### Dependency management
We use [poetry](https://python-poetry.org/) to manage the package and dependencies. 
Follow [their instructions](https://python-poetry.org/docs/#installation) on how 
to install poetry on your system. After that, installing all dependencies is as
simple as 

``` python
poetry install
```

### Virtual environment management
It is good practice to isolate your development environment from other python
installations. There are various tools to do this. We recommend `pyenv` (for 
Linux or MacOS) or [pyenv-win](https://github.com/pyenv-win/pyenv-win) (for 
Windows).


## Style & documentation conventions

We use [ruff](https://github.com/astral-sh/ruff) for linting and code formatting. 
Documentation is based on [mkdocs](https://www.mkdocs.org/)] and [mkdocstringds](https://mkdocstrings.github.io/#:~:text=mkdocstrings.%20Automatic%20documentation%20from%20sources,%20for). Numpy, sphinx and google-style
docstrings are supported, but when writing new code please use [Google-style docstring syntax](https://mkdocstrings.github.io/griffe/docstrings/#google-style)

To support documentation for multiple versions, we use the [mike](https://github.com/jimporter/mike) preprocessor. 
See "How to make a new release" below on how to use mike. 

## Testing

### Locally test 

To run tests locally, use `pytest`:

`poetry run pytest --cov=fm2prof`

To check coverage:

`poetry run pytest --cov=fm2prof`

### Automatic test

Automatic tests are run on each pull request and push. A PR cannot be merged
unless all tests are passing. 

## Deploying

### Locally build executable (optional)

To build a local version of an FM2PROF executable, run:

`poetry run pyinstaller FM2PROF_WINDOWS.spec`

!!! note

    Executables are no longer automatically made since fm2prof was published on pypi 


### How to make a new release

Publishing a new release takes some steps. 

#### Tag your version
After merging a PR to `master`, first make a new tag. Using version `v2.3.0` as an example, 
a tag can be made via the terminal:

``` bash
git tag v2.3.0
```
#### Make a new release

Use Github interface to draft a new release using the appropriate. Document all changes since the previous version. 
If possible, refer to Github Issues. 

#### Update the documentation
We use [mike](https://github.com/jimporter/mike) as a pre-processor for `mkdocs`. To update the documentation
for a new release, use the following line in a terminal:

``` bash
poetry run mike deploy 2.3.0 latest -u
```

This will build the documentation for a specific version (e.g. `2.3.0` for version v2.3.0). The keyword
`latest` will set this to be the default version. The flag `-u` will overwrite any
existing documentation for this versino. 

Finally, you need to push the documentation to github. First, checkout the documentation branch

``` bash
git checkout gh-pages
``` 

and then push changes to github

``` bash 
git push
```

The documentation is hosted on github pages, no other steps are necessary. 




