#!/usr/bin/env python
"""Generate regrid weights for unit testing."""

import ugants.io.load
from esmf_regrid.experimental.io import save_regridder
from esmf_regrid.experimental.unstructured_scheme import GridToMeshESMFRegridder

if __name__ == "__main__":
    source = ugants.io.load.cf("non_ugrid_data.nc")[0]
    target = ugants.io.load.mesh("mesh_C12.nc")
    regridder = GridToMeshESMFRegridder(
        source, target, method="nearest", tgt_location="face"
    )
    save_regridder(regridder, "non_ugrid_data_to_mesh_weights.nc")
