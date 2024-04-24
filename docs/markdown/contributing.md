# Contributing

## Set up development environment

## Style conventions

Please use [Google-style docstring syntax](https://mkdocstrings.github.io/griffe/docstrings/#google-style)

## Test new code

To run tests locally, use pytest:

`poetry run pytest --cov=fm2prof`

To check coverage:

`poetry run pytest --cov=fm2prof`


## Deploying

### Locally build executable

To build a local version of an FM2PROF executable, run:

`poetry run pyinstaller FM2PROF_WINDOWS.spec`

### Making a new release

After merging a PR, make a new tag. Using version `v2.3.0` as an example;

`git tag v2.3.0`

Use Github interface to draft a new release. Attach the executable. Then, update the documentation:

`poetry run mike deploy v2.3.0 latest -u`

Checkout the documentation branch

`git checkout gh-pages`

and push changes to github

`git push`



