# Version history

!!! info "Semantics"
    Version numbering aims to follow [semantic versioning](https://semver.org/).

      * New *patch* versions (e.g. 0.1.**0** to 0.1.**1**) make minor changes that do not alter fuctionality or calculated results.
      * New *minor* versions (e.g. 0.**0**.1 to 0.**1**.0) add new functionality, but will not break your code.  They will not alter the results of calculations with default settings (except for in the hopefully rare case of correcting a bug or typo).
      * New *major* versions (e.g. **0**.1.1 to **1**.0.0) may break your code and require you to rewrite things.  They may significantly alter the results of calculations with default settings.


!!! warning
    The only things that may change, in at least a *minor* version release, are:

      1. Additional features provided with this package, such as for example new functionnalities with new functions to complement the main `canyonb` function. 
      2. Additional code to improve the performance of `canyonb` function: computational time, documentation, etc. 

    The structure of the underlying modules and their functions is not yet totally stable and, for now, may change in any version increment. Such changes will be described in the release notes below.

## 0.4

CONTENT algorithm and multi-dimensional input support.

### 0.4.0 (27 May 2026)

!!! new-version "Changes in v0.4.0"

    ***New features***

    * **`ds.canyonb.content()`** — CONTENT algorithm (Bittig et al. 2018) as an
      xarray accessor method.  Returns internally-consistent AT, CT, pH, and pCO₂
      estimates with full propagated uncertainty, including cross-variable
      covariance from shared hydrographic inputs.
    * **`co2content()`** — public low-level function mirroring `CONTENT.m`.
    * `include_raw=True` exposes the four individual estimates per variable
      (CANYON-B direct + 3 CO2SYS combinations) along an `estimate` dimension.
    * `include_canyonb=True` includes the full CANYON-B sub-output prefixed
      with `canyonb_`.
    * Multi-dimensional datasets (e.g. `time × pressure × lat × lon`) are now
      fully supported by both `predict()` and `content()`.

    ***Bug fixes***

    * **Sigma values** — Jacobians are now computed via forward finite differences
      (signed), matching the `inx` arrays in `CANYONB_private.m`.  The previous
      unit-error approach returned `|inx|`, causing cross-covariances to lose
      their sign and sigma values to be 5–18 % too low.  Values now match the
      MATLAB reference to < 0.1 %.
    * **Dimension ordering** — `DatasetToNumpy._broadcast_all()` now passes
      multi-dimensional variables first to `xr.broadcast`, preserving the
      user-defined dimension order for N-D grids.

    ***Documentation***

    * API autodoc via `mkdocstrings` for `canyonb`, `co2content`,
      `CanyonBAccessor`, and `DatasetToNumpy`.
    * New API pages: *xarray Accessor* and *Preprocessing*.

## 0.3

Native `xarray` support.

### 0.3.0 (2 March 2026)

!!! new-version "Changes in v0.3.0"

    ***New features***

    * **`ds.canyonb.predict()`** — run CANYON-B directly on an `xr.Dataset`
      without any manual extraction.  Results are returned as an `xr.Dataset`
      sharing the same dimensions and coordinates as the input, making it
      trivial to merge predictions back into the original dataset.
    * **`DatasetToNumpy`** — low-level helper class that maps `canyonb`
      argument names to dataset variables (with a customisable `var_map`)
      and converts them to flat numpy arrays.  Useful when you need to
      inspect or pre-process inputs before running the neural network.
    * Both interfaces support a `var_map` parameter for datasets with
      non-standard variable names (e.g. Argo BGC delayed-mode files using
      `TEMP_ADJUSTED`, `PSAL_ADJUSTED`, …).
    * `DatasetToNumpy` is now exported from
      the top-level `canyonbpy` namespace.

    ***Behind the scene***

    * New `canyonbpy/preprocessing.py` module implements `DatasetToNumpy`.
    * `canyonb xarray accessor` lives in `canyonbpy/accessor.py`.
    * New test file `canyonbpy/tests/test_xarray.py` covers both interfaces.

    ***Documentation***

    * Advanced Features page fully rewritten with xarray usage examples,
      including Argo BGC workflows and custom variable name mapping.
    * API reference updated for `DatasetToNumpy`.

## 0.2

Documentation is now available! 

### 0.2.0 (10 January 2025)

!!! new-version "Changes in v0.2.0"

    ***Technical***

    * Online documentation using with `mkdocs` and `readthedocs`. 

    ***Bug fixes***

    * Bug fixes in `README.md`. 

    ***Behind the scene*** 

    * Added inline documentation in Python functions.

## 0.1

Deployment of `canyonbpy` on PyPi! An as-close-as-possible clone of [MATLAB CANYON-B v1.0](https://github.com/HCBScienceProducts/CANYON-B). The version is in production, so will be **1.0** instead of **0.1** pretty soon, but (doing it on my own) I want feedbacks from people before publishing to **1.0.0**. The documentation needs to be done also. 

### 0.1.1 (2 January 2025)

!!! new-version "Changes in v0.1.1"

    ***Bug fixes***
    
    * Updated correct version number. 

    ***Technical***

    * Inline documentation. 
    * Test deployment with multiple Python versions: *3.8, 3.9, 3.10, 3.11*.

### 0.1.0 (2 January 2025)

!!! new-version "Changes in v0.1.1"

    ***New features***
    
    * `canyonb` is available on python!
