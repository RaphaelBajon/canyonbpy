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

<!---
!!! new-version "Changes in v0.1.1"

    ***Breaking changes***

    ***Behind the scene***

    ***New features***

    * 

    ***Default options***

    * 

    ***Bug fixes***
    
    * 

    ***Technical***

    * Updated from building with setup.py to pyproject.toml.
    * canyonbpy can now be installed with conda/mamba (via conda-forge).
-->

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
