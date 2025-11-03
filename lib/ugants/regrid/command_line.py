# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of UG-ANTS and is released under the BSD 3-Clause license.
# See LICENSE.txt in the root of the repository for full licensing details.
"""Implementation for the regrid application."""

import os
from pathlib import Path
from typing import Literal

import esmf_regrid
import numpy as np
from esmf_regrid.experimental.io import load_regridder, save_regridder
from esmf_regrid.experimental.unstructured_scheme import GridToMeshESMFRegridder
from iris.cube import Cube, CubeList
from iris.experimental.ugrid import Mesh, save_mesh
from iris.fileformats.netcdf import save as save_netcdf

from ugants.abc import Application
from ugants.io import save
from ugants.io.load import cf as load_cf
from ugants.io.load import is_netcdf
from ugants.io.save import (
    _check_filepath_extension as verify_output_to_netcdf,
)
from ugants.io.save import (
    ugrid as save_ugrid,
)
from ugants.regrid import band_utils
from ugants.utils.cube import as_cubelist


class Regrid(Application):
    """Regrid regular grid data to an unstructured mesh.

    Parameters
    ----------
    source : :class:`iris.cube.CubeList`
        The regular gridded source data to be regridded.
    target_mesh : :class:`iris.experimental.ugrid.Mesh`
        The UGrid target mesh.
    horizontal_regrid_scheme : :obj:`str`
        The horizontal regrid scheme to be used. Supported schemes are
        "conservative", "bilinear", "nearest".
    tolerance : :obj:`float`
        Tolerance of missing data.
        The value returned in each element of the returned array will be masked if the
        fraction of masked data exceeds this tolerance.
        If not provided, the default tolerance value is zero.
        This option is not available for the "nearest" scheme.
    input_weights
        Path to cached input weights.
        An optional file containing the pre-generated weights for the mesh
        being used for regridding.
    output_weights
        An optional target path for output cached weights. Using pre-cached weights
        makes the regridding process less computationally expensive.



    Raises
    ------
    ValueError
        If a ``tolerance`` is provided and the ``horizontal_regrid_scheme``
        is "nearest".

    Note
    ----
    The data is always regridded to the faces of the unstructured grid.
    """

    results: CubeList = None
    """The source data regridded onto the faces of the target mesh."""
    source: CubeList = None
    """The source data to be regridded."""
    target_mesh: Mesh = None
    """The mesh to regrid to."""

    _loader = load_cf

    def __init__(
        self,
        source: CubeList,
        target_mesh: Mesh,
        horizontal_regrid_scheme: Literal["conservative", "bilinear", "nearest"],
        tolerance: float = 0,
        input_weights: str = "",
        output_weights: str = "",
    ):
        if tolerance and horizontal_regrid_scheme == "nearest":
            raise ValueError(
                "The 'tolerance' option is not available for regrid scheme 'nearest'"
            )
        source = as_cubelist(source)
        if len(source) > 1:
            _validate_source_grids(source)
        if input_weights != "" and output_weights != "":
            raise ValueError(
                "Only one of input_weights and output_weights can be provided"
            )
        self.source = source
        self.target_mesh = target_mesh
        self.horizontal_regrid_scheme = horizontal_regrid_scheme
        self.tolerance = tolerance
        self.input_weights = input_weights
        self.output_weights = output_weights

    def run(self):
        """Regrid :attr:`source` to :attr:`target_mesh`.

        The result of the regrid is stored in :attr:`results`.

        """
        source_cube = self.source[0]
        if self.input_weights:
            is_netcdf(self.input_weights)
            self.regridder = load_regridder(Path(self.input_weights))
            _validate_input_weights(
                self.regridder,
                self.tolerance,
                self.horizontal_regrid_scheme,
            )
        else:
            self.regridder = GridToMeshESMFRegridder(
                source_cube,
                self.target_mesh,
                method=self.horizontal_regrid_scheme,
                tgt_location="face",
                mdtol=self.tolerance,
            )
        self.results = CubeList(self.regridder(cube) for cube in self.source)

    def save(self):
        """Save ``self.results`` to ``self.output``."""
        verify_output_to_netcdf(Path(self.output))
        save_ugrid(self.results, self.output)

        if self.output_weights:
            save_regridder(self.regridder, str(self.output_weights))


def _validate_input_weights(
    regridder: GridToMeshESMFRegridder,
    tolerance: float,
    horizontal_regrid_scheme: Literal["conservative", "bilinear", "nearest"],
):
    if regridder.mdtol != tolerance:
        raise ValueError(
            "Tolerance value of input_weights does not match the value"
            " provided on command line."
        )
    if regridder.method != horizontal_regrid_scheme:
        raise ValueError(
            "Regrid scheme of input_weights does not match the scheme "
            "provided on command line."
        )


class RegridMeshToMesh(Application):
    """Regrid unstructured mesh data to an unstructured mesh of a different resolution.

    Parameters
    ----------
    source
        The unstructured mesh data to be regridded.
    target_mesh
        The UGrid target resolution mesh.
    horizontal_regrid_scheme
        The horizontal regrid scheme to be used. Supported schemes are
        "conservative", "bilinear", "nearest".
    tolerance
        Tolerance of missing data.
        The value returned in each element of the returned array will be masked if the
        fraction of masked data exceeds this tolerance.
        If not provided, the default tolerance value is zero.
        This option is not available for the "nearest" scheme.

    Raises
    ------
    ValueError
        If a ``tolerance`` is provided and the ``horizontal_regrid_scheme``
        is "nearest".

    Note
    ----
    The data is always regridded to the faces of the unstructured mesh.
    """

    results: CubeList = None
    """The source data regridded onto the faces of the target mesh."""

    def __init__(
        self,
        source: CubeList,
        target_mesh: Mesh,
        horizontal_regrid_scheme: Literal["conservative", "bilinear", "nearest"],
        tolerance: float = 0,
    ):
        if tolerance and horizontal_regrid_scheme == "nearest":
            raise ValueError(
                "The 'tolerance' option is not available for regrid scheme 'nearest'"
            )
        source = as_cubelist(source)
        if len(source) > 1:
            _validate_source_meshes(source)
        self.source = source
        self.target_mesh = target_mesh
        self.horizontal_regrid_scheme = horizontal_regrid_scheme
        self.tolerance = tolerance

    def run(self):
        """Regrid ``self.source`` to ``self.target_mesh``."""
        regridder_lookup = {
            "conservative": esmf_regrid.ESMFAreaWeightedRegridder,
            "bilinear": esmf_regrid.ESMFBilinearRegridder,
            "nearest": esmf_regrid.ESMFNearestRegridder,
        }
        regridder_kwargs = {
            "src": self.source[0],
            "tgt": self.target_mesh,
            "tgt_location": "face",
        }
        # The nearest neighbour scheme does not accept a mdtol argument
        # but other schemes do
        if self.horizontal_regrid_scheme != "nearest":
            regridder_kwargs["mdtol"] = self.tolerance

        # Select the appropriate regridder from the lookup, and instantiate with
        # appropriate keyword arguments. This regridder can then be reused for all cubes
        regridder = regridder_lookup[self.horizontal_regrid_scheme](**regridder_kwargs)
        self.results = CubeList(regridder(source) for source in self.source)


class SplitGridToMeshByLatitude(Application):
    """Split the provided regular gridded source and target mesh into latitude bands.

    Parameters
    ----------
    source : :class:`~iris.cube.CubeList`
        The global, regular gridded source data to be split.
    target_mesh : :class:`~iris.experimental.ugrid.Mesh`
        The target mesh to be split.
    number_of_bands : int
        Number of latitude bands to split by.
    """

    _loader = load_cf

    results: list[CubeList] = None
    """A list of CubeLists, one for each latitude band.
    Each CubeList contains the same number of cubes as the source CubeList.
    The domain of each cube completely covers the corresponding mesh band in
    :attr:`mesh_bands`. Padding is added ensure that the mesh band is
    fully enclosed, so there will be overlap between adjacent source bands.
    See :func:`~ugants.regrid.band_utils.cube_subset_latitude_bounds` for more details
    on how the padding is calculated.
    An attribute ``band_number`` is added to distinguish each cube."""

    mesh_bands: list[Mesh]
    """Each mesh covers a latitude band, approximately evenly spaced.
    There are no overlapping cells between any two mesh bands, so each mesh
    band covers a unique region. Together, the mesh bands cover the entire
    target mesh."""

    mesh_mapping_cube: Cube
    """A UGrid cube constructed from the ``target_mesh``. The data maps each cell
    to its corresponding latitude band. For example, if a cell has a value of 1,
    then it belongs in latitude band 1."""

    output: str = None
    """The output **directory** to which to write the
    :attr:`results`, :attr:`mesh_bands` and
    :attr:`mesh_mapping_cube`.

    """

    def __init__(self, source: CubeList, target_mesh: Mesh, number_of_bands: int):
        source = as_cubelist(source)
        if len(source) > 1:
            _validate_source_grids(source)
        _validate_source_is_global(source[0])
        _validate_number_of_bands(number_of_bands)
        self.source = source
        self.target_mesh = target_mesh
        self.number_of_bands = number_of_bands

    def run(self):
        """Run the application.

        The source and target are split into bands of approximately equal latitude.

        There is **no** overlap between target bands, i.e. a cell in the original target
        mesh will appear in one target band only. The mapping between target
        mesh cells and band number is recorded in the :attr:`mesh_mapping_cube`.

        There **is** overlap between source bands, i.e. a cell in the original source
        may appear in more than one source band. This is because the source domain must
        extend beyond the target domain in order to capture all the required data for
        regridding.

        The following attributes are set by this method:

        * :attr:`results`
        * :attr:`mesh_bands`
        * :attr:`mesh_mapping_cube`

        """
        source_cube = self.source[0]

        target_min_latitude = min(self.target_mesh.node_coords.node_y.points)
        target_max_latitude = max(self.target_mesh.node_coords.node_y.points)
        band_bounds = band_utils.generate_band_bounds(
            start=target_min_latitude,
            stop=target_max_latitude,
            n_bands=self.number_of_bands,
        )
        self.mesh_mapping_cube = band_utils.mesh_to_cube(self.target_mesh)

        # A boolean array of shape (n_bands, n_faces)
        mesh_indices_per_band = np.array(
            [
                band_utils.find_cell_centres_within_latitude_bounds(
                    self.mesh_mapping_cube, bounds
                )
                for bounds in band_bounds
            ],
        )
        # An integer array of shape (n_faces,) labelling each face according to its band
        self.mesh_mapping_cube.data = mesh_indices_per_band.argmax(0)

        self.mesh_bands = [
            band_utils.subset_mesh_cube_by_indices(self.mesh_mapping_cube, indices).mesh
            for indices in mesh_indices_per_band
        ]
        # A list of band bounds tuples (lower_latitude, upper_latitude),
        # of length n_bands
        latitude_bounds_for_source = [
            band_utils.cube_subset_latitude_bounds(
                subsetted_mesh, source_cube.coord("latitude")
            )
            for subsetted_mesh in self.mesh_bands
        ]

        self.results = []
        # 1. iterate over latitude bands
        for band_number, band_bounds in enumerate(latitude_bounds_for_source):
            latitude_band_cubelist = CubeList()
            # 2. iterate over source cubes, extract latitude band from each source cube
            for source_cube in self.source:
                latitude_band_cube = band_utils.constrain_source_cube_latitude(
                    source_cube, band_bounds
                )
                latitude_band_cube.attributes["band_number"] = band_number
                latitude_band_cubelist.append(latitude_band_cube)
            self.results.append(latitude_band_cubelist)

    def save(self):
        """Save the latitude bands to NetCDF.

        Three types of file are saved to the directory specified by ``self.output``:

        * :attr:`results`: ``number_of_bands`` such files are output, named
          ``source_band_{band_number}.nc``.
        * :attr:`mesh_bands`: ``number_of_bands`` such files are output, named
          ``mesh_band_{band_number}.nc``.
        * :attr:`mesh_mapping_cube`: only one such file is output, named
          ``mesh_band_mapping.nc``.

        In total, :code:`2*number_of_bands + 1` files are output.
        """
        if self.output is None:
            raise ValueError("No output directory location has been set.")

        if not hasattr(self, "mesh_mapping_cube"):
            raise ValueError(
                "The application has not yet been run, mesh_mapping_cube is not set."
            )

        if not hasattr(self, "mesh_bands"):
            raise ValueError(
                "The application has not yet been run, mesh_bands is not set."
            )

        if self.results is None:
            raise ValueError(
                "The application has not yet been run, results is not set."
            )

        save.ugrid(
            self.mesh_mapping_cube, os.path.join(self.output, "mesh_band_mapping.nc")
        )
        for band_number, result, mesh_band in zip(
            range(len(self.results)),
            self.results,
            self.mesh_bands,
            strict=True,
        ):
            save_mesh(
                mesh_band, os.path.join(self.output, f"mesh_band_{band_number}.nc")
            )
            save_netcdf(
                result,
                os.path.join(self.output, f"source_band_{band_number}.nc"),
            )


def _validate_source_grids(source: CubeList):
    """Check that all cubes have the same horizontal grid.

    Does not check dimension ordering, only that the horizontal coordinates are equal.

    Parameters
    ----------
    source: iris.cube.CubeList
        Cubes to be compared. Must be at least 2 cubes.

    Raises
    ------
    ValueError
        If any pair of cubes have different x or y coordinates.
    """
    reference_grid_x = source[0].coord(axis="x")
    reference_grid_y = source[0].coord(axis="y")
    for cube in source[1:]:
        if (cube.coord(axis="x") != reference_grid_x) or (
            cube.coord(axis="y") != reference_grid_y
        ):
            raise ValueError("Not all source cubes have the same horizontal grid.")


def _validate_source_meshes(source: CubeList):
    """Check that all cubes have the same horizontal mesh.

    Parameters
    ----------
    source: iris.cube.CubeList
        Cubes to be compared. Must be at least 2 cubes.

    Raises
    ------
    ValueError
        If any pair of cubes have different meshes.
    """
    for cube in source[1:]:
        if cube.mesh != source[0].mesh:
            raise ValueError("Not all source cubes have the same horizontal mesh.")


def _validate_source_cubelist_length(source_cubelist: CubeList):
    """Check that the source cubelist contains only one cube.

    Parameters
    ----------
    source_cubelist : CubeList
        The source cubelist to validate

    Raises
    ------
    ValueError
        If there is not only one cube in the source cubelist.
    """
    number_source_cubes = len(source_cubelist)
    if number_source_cubes != 1:
        raise ValueError(f"Source contained {number_source_cubes} cubes, expected 1.")


def _validate_source_is_global(source_cube: Cube):
    """Check that the source cube is global.

    The following checks are performed on the cube's horizontal coordinates:

    * Longitude (axis="x") must be circular.
    * Latitude (axis="y") bounds must extend from -90 to +90 degrees.

    Parameters
    ----------
    source_cube : Cube
        The source cube to validate

    Raises
    ------
    ValueError
        If the source data is not global
    """
    longitude = source_cube.coord(axis="x").copy()
    if not longitude.circular:
        raise ValueError(
            "The provided source is not global: longitude is not circular."
        )

    latitude = source_cube.coord(axis="y").copy()
    if not latitude.has_bounds():
        latitude.guess_bounds()

    min_lat = latitude.bounds.min()
    max_lat = latitude.bounds.max()
    if (min_lat, max_lat) != (-90.0, 90.0):
        raise ValueError(
            f"The provided source is not global: latitude min = {min_lat}, "
            f"latitude max = {max_lat}"
        )


def _validate_number_of_bands(number_of_bands: int):
    """Check that the number of bands is greater than one.

    Parameters
    ----------
    number_of_bands : int
        The number of bands to attempt to split by.

    Raises
    ------
    ValueError
        If the number of bands is not greater than one.
    """
    if number_of_bands < 2:
        raise ValueError(
            f"The number of bands must be greater than 1, got {number_of_bands}."
        )


class RecombineMeshBands(Application):
    """Recombine regridded latitude bands into a single cube.

    Parameters
    ----------
    mesh_mapping : iris.cube.CubeList
        A single-element CubList which maps individual latitude bands to cells
        in the target mesh. See also
        :attr:`SplitGridToMeshByLatitude.mesh_mapping_cube`.
    bands : iris.cube.CubeList
        The regridded latitude bands to recombine into a single UGrid cube.
        Each band cube must have a ``band_number`` attribute, which maps the
        band to its location on the ``mesh_mapping``.

    """

    results: CubeList = None
    """The recombined bands in a single cube. The cube's mesh is identical
    to that of the mesh mapping, and the data is taken from the cubes in ``bands``."""

    def __init__(self, mesh_mapping: CubeList, bands: CubeList):
        self.mesh_mapping = as_cubelist(mesh_mapping)
        self.bands = bands
        self.names = {cube.name() for cube in self.bands}

        # Validate that the mesh mapping has a single unstructured dimension
        if self.mesh_mapping[0].mesh is None:
            raise ValueError("The provided mesh_mapping does not contain a mesh.")
        if (ndim := self.mesh_mapping[0].ndim) != 1:
            raise ValueError(
                f"The provided mesh_mapping should have 1 dimension, got {ndim}."
            )

        # Validate that the expected numbers of bands are provided for each variable
        # Validate consistency in unstructured mesh dimension length
        expected_max_band_number = int(self.mesh_mapping[0].data.max())
        expected_band_numbers = list(range(expected_max_band_number + 1))
        self._target_mesh_dim_length = self.mesh_mapping[0].shape[0]
        for name in self.names:
            band_numbers_by_name = sorted(
                cube.attributes["band_number"] for cube in self.bands.extract(name)
            )
            if band_numbers_by_name != expected_band_numbers:
                raise ValueError(
                    f"Inconsistent mesh bands provided for {name}: expected "
                    f"{expected_band_numbers}, got {band_numbers_by_name}"
                )

            # Sum of the number of points in the mesh dimension of each cube
            bands_mesh_dim_lengths = [
                cube.shape[cube.mesh_dim()] for cube in self.bands.extract(name)
            ]
            bands_mesh_total_dim_length = sum(bands_mesh_dim_lengths)
            if bands_mesh_total_dim_length != self._target_mesh_dim_length:
                raise ValueError(
                    f"Inconsistent unstructured dimension lengths for {name} bands. "
                    "Provided bands have unstructured dimensions of lengths "
                    f"{bands_mesh_dim_lengths} giving a total of "
                    f"{bands_mesh_total_dim_length}, whereas mesh mapping has length "
                    f"{self._target_mesh_dim_length}."
                )

    def run(self):
        """Recombine the latitude bands into a single cube.

        The data from each latitude band cube in the ``bands``
        CubeList are used to fill the corresponding cells in the target mesh,
        according to the band cube's ``band_number`` attribute.

        The ``mesh_mapping_cube`` provides the target mesh to be filled with data from
        the regridded latitude bands. The data in ``mesh_mapping_cube`` describe which
        latitude band is to be used to fill the cells.

        For example, cells with value 2 in the ``mesh_mapping_cube``
        are filled with data from band number 2.

        This method sets the :attr:`results` attribute.
        """
        self.results = CubeList(
            self._recombine_single_variable(
                self.bands.extract(name), self.mesh_mapping[0]
            )
            for name in self.names
        )

    def _recombine_single_variable(self, regrid_bands: CubeList, mesh_mapping: Cube):
        reference_band = regrid_bands[0]
        target_shape = list(reference_band.shape)
        mesh_dim = reference_band.mesh_dim()
        target_shape[mesh_dim] = self._target_mesh_dim_length
        target_array = np.ma.masked_all(target_shape, dtype=reference_band.dtype)

        for regrid_band in regrid_bands:
            band_number = regrid_band.attributes["band_number"]
            indices_in_band = np.nonzero(mesh_mapping.data == band_number)[0]
            slice_to_fill = [slice(None)] * target_array.ndim
            slice_to_fill[mesh_dim] = indices_in_band
            target_array[tuple(slice_to_fill)] = regrid_band.data

        dim_coords_and_dims = [
            (coord, reference_band.coord_dims(coord))
            for coord in reference_band.dim_coords
        ]
        aux_coords_and_dims = [
            (coord, reference_band.mesh_dim()) for coord in mesh_mapping.aux_coords
        ]

        result = Cube(
            target_array,
            dim_coords_and_dims=dim_coords_and_dims,
            aux_coords_and_dims=aux_coords_and_dims,
        )
        result.metadata = reference_band.metadata
        result.attributes.pop("band_number")
        return result
