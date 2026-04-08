<!-- Some of the content of this file has been produced with the assistance of Met Office Github Copilot Enterprise. -->

## Project overview

This project is for the core library for generating ancillary files for LFRic on a UGrid mesh.  It is built on top of iris, and relies on [ANTS](https://github.com/MetOffice/ANTS) for any pre-processing needed on the regular source grid.

Science applications for generating UGrid ancillaries for Momentum may be found in the published [working practices](https://metoffice.github.io/ancil-working-practices) or in the corresponding [repository](https://github.com/MetOffice/ug-ancillary-file-science).

netCDF is the only supported data format.

## Code style

All code should follow the [working practices](https://github.com/MetOffice/ancil-working-practices).

## CLA

All code contributors need to sign the CLA.  Under no circumstances can an AI agent sign the CLA.

## Directory structure

The directory structure is as follows:

```
bin/ - generic ancillary generation applications in python.
docs/ - static documentation and documentation configuration in rst.
lib/ugants/ - library code for generating UGrid ancillaries in python.
lib/ugants/tests/ - unit tests for the library code in python following pytest style.
rose-stem/ - cylc workflow for end to end testing.
utils/ - infrastructure for the end to end testing in python.
```


