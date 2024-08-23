# Contributing to TX2

Help in improving DASP-Stacker is welcome!

If you find a bug or think of an enhancement/improvement you would like to see,
feel free to fill out an appropriate
[issue](https://github.com/ORNL/dasp-stacker/issues/new).

If you have a question, double check that it's not covered in the function documentation found in the[readme](https://github.com/ORNL/dasp-stacker/blob/main/README.md)

The notebooks in the [examples](https://github.com/ORNL/dasp-stacker/tree/main/examples) directory show the intended use of the functionality provided by the library.

## Submitting a PR

If you have added a useful feature or fixed a bug, open a new pull request with
the changes.  When submitting a pull request, please describe what the pull
request is addressing and briefly list any significant changes made. If it's in
regards to a specific issue, please include the issue number. Please check and
follow the formatting conventions below!

## Code Formatting

This project uses the [black code formatter](https://github.com/psf/black).

Any public functions and classes should be clearly documented with
[google-style docstrings](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html).

## Development Setup

```bash
mamba env create -f environment.yaml
pre-commit install
```
