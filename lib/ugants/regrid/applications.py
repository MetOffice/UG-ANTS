# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of UG-ANTS and is released under the BSD 3-Clause license.
# See LICENSE.txt in the root of the repository for full licensing details.
"""Implementation for the mesh to grid regrid application."""

import os
from pathlib import Path
from typing import Literal

import iris
import iris.fileformats.netcdf
from esmf_regrid.experimental.io import load_regridder, save_regridder
from esmf_regrid.experimental.unstructured_scheme import (
    MeshToGridESMFRegridder,
)
from iris.cube import CubeList
from iris.experimental.ugrid import PARSE_UGRID_ON_LOAD, Mesh
from iris.fileformats.netcdf import save as save_netcdf

from ugants.abc import Application
from ugants.io import save
from ugants.io.load import cf as load_cf
from ugants.regrid import band_utils
from ugants.regrid.command_line import (
    _validate_number_of_bands,
    _validate_source_cubelist_length,
)
from ugants.utils.cube import prepare_for_save


def _load_any_cubes(uri):
    with PARSE_UGRID_ON_LOAD.context():
        cubelist = iris.load(uri)
    return cubelist


class MeshToGridRegrid(Application):
    """Regrid unstructured grid data to regular lat lon.

    Parameters
    ----------
    source :
        The ugrid source data.
    target_grid :
        The regular gridded target grid.
    horizontal_regrid_scheme :
        The horizontal regrid scheme to be used. Supported schemes are
        "conservative", "bilinear", "nearest".
    input_weights
        Path to cached input weights.
        An optional file containing the pre-generated weights for the mesh
        being used for regridding.
    output_weights
        An optional target path for output cached weights. Using pre-cached weights
        makes the regridding process less computationally expensive.
    """

    _loader = _load_any_cubes

    def __init__(
        self,
        source: CubeList,
        target_grid: CubeList,
        horizontal_regrid_scheme: Literal["conservative", "bilinear", "nearest"],
        input_weights: str = "",
        output_weights: str = "",
    ):
        if input_weights != "" and output_weights != "":
            raise ValueError(
                "Only one of input_weights and output_weights can be provided"
            )
        self.source = source
        self.target_grid = target_grid
        self.horizontal_regrid_scheme = horizontal_regrid_scheme
        self.input_weights = input_weights
        self.output_weights = output_weights

    def run(self):
        """Regrid ``self.source`` to ``self.target_grid``."""
        number_source_cubes = len(self.source)
        if number_source_cubes != 1:
            raise ValueError(
                f"Source contained {number_source_cubes} cubes, expected 1."
            )
        number_target_cubes = len(self.target_grid)

        if number_target_cubes != 1:
            raise ValueError(
                f"Target Grid contained {number_target_cubes} cubes, expected 1."
            )
        source_cube = self.source[0]
        target_cube = self.target_grid[0]

        if self.input_weights:
            regridder = load_regridder(Path(self.input_weights))
            if regridder.method != self.horizontal_regrid_scheme:
                raise ValueError(
                    "Regrid scheme of input_weights does not match the scheme "
                    "provided on command line."
                )
        else:
            regridder = MeshToGridESMFRegridder(
                source_cube,
                target_cube,
                method=self.horizontal_regrid_scheme,
            )

        self.regridder = regridder
        self.results = regridder(source_cube)

    def save(self):
        """Save ``self.results`` to ``self.output``."""
        results = prepare_for_save(self.results)
        save_netcdf(results, self.output)
        if self.output_weights:
            save_regridder(self.regridder, str(self.output_weights))


class SplitMeshToGridByLatitude(Application):
    """Split the provided UGrid source and regular grid target into latitude bands.

    Parameters
    ----------
    source : :class:`~iris.cube.CubeList`
        The global UGrid source data to be split.
    target : :class:`~iris.experimental.ugrid.Mesh`
        The regular grid target to be split.
    number_of_bands : int
        Number of latitude bands to split by.
    """

    source_bands: CubeList
    """The domain of each cube completely covers the corresponding band in
    :attr:`target_bands`.  An attribute ``band_number`` is added to
    distinguish each cube."""

    target_bands: list[Mesh]
    """Each target band covers a latitude band, approximately evenly spaced.
    There are no overlapping cells between any two bands, so each band covers
    a unique region. Together, the bands cover the entire target domain."""

    output: str = None
    """The output **directory** to which to write the :attr:`source_bands` and
    :attr:`target_bands`."""

    # TODO: https://github.com/MetOffice/UG-ANTS/issues/43
    #  Replace with a central load cube when that is available.
    _loader = _load_any_cubes

    def __init__(self, source: CubeList, target: CubeList, number_of_bands: int):
        _validate_source_cubelist_length(source)
        try:
            _validate_source_cubelist_length(target)
        except ValueError:
            raise ValueError(
                f"Target contained {len(target)} cubes, expected 1."
            ) from None
        _validate_number_of_bands(number_of_bands)

        self.source = source[0]
        self.target = target[0]
        self.number_of_bands = number_of_bands

    def run(self):
        """Split the target and source cubes into bands.

        The target is first split into bands by latitude, with the resulting
        bands stored in :attr:`target_bands`.  These bands are then used to
        calculate the source bands, where each source band fully contains the
        corresponding target band.  The source bands are stored in
        :attr:`source_bands`.

        """
        self.source_bands = CubeList()
        self.target_bands = band_utils.split_cube(self.target, self.number_of_bands)

        for target_band_cube in self.target_bands:
            target_band_bounds = band_utils.cube_latitude_bounds(target_band_cube)
            source_band_indices = band_utils.get_faces_that_overlap_bounds(
                self.source, target_band_bounds
            )
            source_band_cube = band_utils.subset_mesh_cube_by_indices(
                self.source, source_band_indices
            )

            self.source_bands.append(source_band_cube)

    def save(self):
        """Save source and target bands.

        Saves all the source and target bands to individual netCDF files.
        Source bands are saved to the :attr:`output` directory, with a name of
        ``source_band_N.nc`` for the Nth source band.  The corresponding Nth
        target band is saved in the same directory with the name
        ``target_band_N.nc``.

        """
        for band_number, source_band, target_band in zip(
            range(len(self.source_bands)),
            self.source_bands,
            self.target_bands,
            strict=True,
        ):
            save.ugrid(
                source_band,
                os.path.join(self.output, f"source_band_{band_number}.nc"),
            )
            iris.fileformats.netcdf.save(
                target_band,
                os.path.join(self.output, f"target_band_{band_number}.nc"),
            )


class RecombineGridBands(Application):
    """Combines individual split results into a single result.

    Parameters
    ----------
    slices : iris.cube.CubeList
        Paths to files containing individual results.
    """

    _loader = load_cf

    def __init__(self, slices: CubeList):
        self.slices = slices

    def run(self):
        """Recombine individual split results into a complete result.

        Sets the :attr:`~ugants.abc.Application.results` to the complete
        result.  If the individual split results have different data types,
        this promotes the data type of the result to the highest precision
        data type in the split results.  The ``history`` attribute of the
        combined result is set to be the history of the first individual band.

        """
        self.results = self.slices.copy()
        history = self.results[0].attributes["history"]
        for cube in self.results:
            cube.attributes["history"] = history
        try:
            self.results = self.results.concatenate_cube()
        except iris.exceptions.ConcatenateError:
            self._reconcile_data_type()
            self.results = self.results.concatenate_cube()

    def _reconcile_data_type(self):
        """Casts all cubes in :attr:`results` to the same data type.

        The data type used is the data type of the highest precision
        cube in :attr:`results`.

        Works for lazy and realised data.
        """
        # iris-esmf-regrid will return float32 in some cases and float64
        # in others.  We'll scale everything up to the larger data type
        # and re-concatenate:
        largest_itemsize = 0
        for cube in self.results:
            data = cube.core_data()
            if kind := data.dtype.kind != "f":
                raise ValueError(
                    f"Expected floating point data, got numpy dtype kind of {kind}"
                )
            if data_itemsize := data.dtype.itemsize > largest_itemsize:
                largest_itemsize = data_itemsize
                output_dtype = data.dtype
        # Dask and numpy both have `astype`; and since the dtype is being
        # taken from the cube, this is sufficient to avoid needing special
        # case code for lazy or realised data:
        for cube in self.results:
            cube.data = cube.core_data().astype(output_dtype)

    def save(self):
        """Save recombined grid bands to sinlge output file.

        Specifically, saves recombined :attr:`~ugants.abc.Application.results`
        to :attr:`~ugants.abc.Application.output`.

        """
        iris.fileformats.netcdf.save(self.results, self.output)
