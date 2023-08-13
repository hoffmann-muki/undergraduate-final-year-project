### py-gpfa

This is a python port of the MATLAB code for Gaussian Process Factor Analysis (GPFA) that was made
available by Yu et al. at [GPFA_MATLAB_CODE](http://users.ece.cmu.edu/~byronyu/software/gpfa0203.tgz)

### To run
> ``python example.py``

A command line interface will guide the user to select a kernel, pick a neural recording, and set up any necessary GPFA parameters.

### Dependencies

- numpy >= 1.14

- scipy >= 1.1

- For LaTeX typesetting in the plots, some TeX libraries are required. Try installing with [MacTeX](http://www.tug.org/mactex/morepackages.html) (for MacOS) and some more packages with TeX Live Manager:
> ``tlmgr install dvipng type1cm``
