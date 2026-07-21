# Core API Reference

::: canyonbpy.core.canyonb
    options:
      show_root_heading: true
      show_source: true

---

::: canyonbpy.content.co2content
    options:
      show_root_heading: true
      show_source: true

<!-- Legacy hand-written reference kept below for completeness -->


!!! info "Some final notes on the core API"

    * NaN values in inputs should result in NaN predictions
    * Arctic latitudes are automatically adjusted
    * Time can be provided as `datetime`, `datetime64`/`Timestamp`, date
      strings, or decimal years — see [Input Data](../user-guide/input.md)