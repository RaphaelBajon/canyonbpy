# About

`canyonbpy` is maintained primarily by R. Bajon of Ifremer (France). It is a Python package to use the Matlab CANYON-B function developed by [Bittig et al., 2018](https://doi.org/10.3389/fmars.2018.00328). It was developed from the MATLAB [CANYON-B v1.0](https://github.com/HCBScienceProducts/CANYON-B) in date of 11.09.2018, corresponding to the initial publication of the paper.

## Stars

If you like or use this package, please give a star on [Github](https://github.com/RaphaelBajon/canyonbpy).

## Citation

If you use `canyonbpy` in your research, please cite both the original CANYON-B paper and the current Python implementation. Each release is archived on [Zenodo](https://zenodo.org) with a versioned DOI — cite the specific version you used for full reproducibility.

**Original CANYON-B paper:**

```bibtex
@article{bittig2018canyon,
  title     = {An alternative to static climatologies: Robust estimation of open ocean CO2 variables and nutrient concentrations from T, S, and O2 data using Bayesian neural networks},
  author    = {Bittig, Henry C and Steinhoff, Tobias and Claustre, Hervé and Körtzinger, Arne and others},
  journal   = {Frontiers in Marine Science},
  volume    = {5},
  pages     = {328},
  year      = {2018},
  publisher = {Frontiers},
  doi       = {10.3389/fmars.2018.00328}
}
```

**This package (latest version):**

```bibtex
@software{bajon_canyonbpy,
  author    = {Bajon, Raphaël},
  title     = {canyonbpy: A Python implementation of CANYON-B},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.18868524},
  url       = {https://doi.org/10.5281/zenodo.18868524}
}
```

!!! tip "Cite a specific version"
    Each release has its own versioned DOI on Zenodo. You can find the DOI for the
    exact version you used on the [Zenodo record](https://doi.org/10.5281/zenodo.14765787)
    or in the release notes on [GitHub](https://github.com/RaphaelBajon/canyonbpy/releases).

    Example for v0.3.0:
    ```bibtex
    @software{bajon_canyonbpy_v030,
      author    = {Bajon, Raphaël},
      title     = {canyonbpy: A Python implementation of CANYON-B},
      version   = {0.3.0},
      publisher = {Zenodo},
      doi       = {10.5281/zenodo.18868524}
    }
    ```
