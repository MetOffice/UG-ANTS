# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of UG-ANTS and is released under the BSD 3-Clause license.

# See LICENSE.txt in the root of the repository for full licensing details.
"""Generic support code for defining Mesh filtering objects."""
import abc

from iris.cube import Cube
from iris.experimental.ugrid.mesh import Mesh


class UnstructuredFilterABC(abc.ABC):
    """
    An abstract class from which specific types of mesh filter are derived.

    Represents a spatial operation defined on a given unstructured mesh, provided at
    creation time.  The operation itself will be defined by the inheriting subclass.

    The general operation of this will consist of several steps :

    * creation via :meth:`~ugants.filter.generic.UnstructuredFilterABC.__init__` :
      e.g. ``filter = FilterClass(cube[, *args, **kwargs])``

      The initial `__init__`, i.e. constructor call : this specifies and records
      the target mesh on which the filter operates.  Additional control arguments and
      keywords may also be provided, but these will be specific to the actual filter
      type (class).

    * call via :meth:`~ugants.filter.generic.UnstructuredFilterABC.__call__` :
      e.g. ``result_cube = filter(input_cube)``

      The actual filtering operation.  The input_cube, and the result, contain data on
      the same, originally-specifed filter mesh and location.
      The input cube can have different dimensions to the original, but its mesh
      (and hence mesh dimension length) must be the same.  The calculation will be
      repeated over non-mesh dimensions, and the result will have the same dimensions,
      and same mesh, as the input.

    Notes
    -----
    In defining a specific derived subclass, you define the
    :meth:`UnstructuredFilterABC.inner_filter` method, which is the core calculation.
    However, it will usually also be useful to override the '__init__' method.
    When providing an overriding '__init__' method, it is essential that this also
    calls the parent :meth:`UnstructuredFilterABC.__init__`.
    """

    def __init__(self, cube):
        """
        Create a filter object.

        Parameters
        ----------
        cube : :class:`~iris.cube.Cube`
            The Iris :class:`~iris.cube.Cube` or :class:`~iris.experimental.ugrid.Mesh`
            defining the mesh, and location, which the operation operates on.
            Only its `.mesh` and `.location` are relevant to the filter operation.

        Notes
        -----
        The input mesh+location are stored for this filter, along with any other
        (filter-type-specific) args and keywords.  These apply to all subsequent
        operations.

        """
        if not isinstance(cube, Cube):
            message = (
                "First argument of UnstructuredFilterABC must be a Cube : "
                f'got "{cube!r}".'
            )
            raise TypeError(message)

        #: The mesh on which the filter operation is defined.
        self.mesh: Mesh = cube.mesh
        if not self.mesh:
            message = (
                f'Cannot base an UnstructuredFilterABC on cube "{cube.name()}", '
                "as it has no mesh."
            )
            raise ValueError(message)

        #: The mesh location on which the filter operation is defined.
        self.location: str = cube.location
        #: Whether the precalculation was already done.
        self.precalculation_done: bool = False

    def _validate_call_argument(self, cube):
        """Check that the operation input matches requirements."""
        if not isinstance(cube, Cube):
            message = f'First argument must be a Cube, got : "{type(cube)}".'
            raise TypeError(message)
        if cube.mesh != self.mesh:
            message = (
                f'Input cube mesh "{cube.mesh!r}" does not match the original filter '
                f'mesh, "{self.mesh!r}".'
            )
            raise ValueError(message)
        if cube.location != self.location:
            message = (
                f'Input cube location "{cube.location}" does not match the original '
                f'filter location, "{self.location}".'
            )
            raise ValueError(message)

    def _validate_inner_filter_result(self, result, cube):
        """Check that various properties of the returned result are as required."""
        if not isinstance(result, Cube):
            message = (
                f"Inner-filter call returned a value which is not a Cube : {result}."
            )
            raise TypeError(message)

        for aspect in ("shape", "mesh", "location"):
            input_aspect = getattr(cube, aspect)
            result_aspect = getattr(result, aspect)
            if result_aspect != input_aspect:
                message = (
                    f"Inner-filter call returned a cube of a different {aspect} to "
                    "the input : "
                    f"{result_aspect!r} != {input_aspect!r}."
                )
                raise ValueError(message)

    def __call__(self, cube):
        """Perform an actual filter calculation.

        Parameters
        ----------
        cube : :class:`~iris.cube.Cube`
            the input cube : it must have the expected mesh and mesh-location.

        Raises
        ------
        TypeError
            If the input cube or the result of the filter is not a
            :class:`~iris.cube.Cube`.
        ValueError
            If there is an inconsistency between the cube used to create a
            filter, the cube on which the filter is being called, and
            the cube returned by the filtering operation.

        Returns
        -------
        :class:`iris.cube.Cube`
            A new cube, on the original mesh, with all other dimensions and components
            derived from the input cube.
            Unless the filter operation assigns a specific standard-name, it will not
            have one.
            The calculation is simply repeated over non-mesh dimensions, it cannot
            depend on them.

        """
        self._validate_call_argument(cube)

        # Create an input cube copy, but do not replicate the data array.
        # N.B. this will make copies of aux-coords etc, but *not* the Mesh.
        inner_cube = cube.copy(cube.core_data())

        # Remove the standard-name, since the original phenomenon identity may not be
        # correct for the result, and we can't define standard-name or cell-methods.
        # This is "safe", but the inner method may re-install a suitable standard-name
        # and/or cell methods, units etc.
        original_standard_name = cube.standard_name
        inner_cube.standard_name = None

        # Call the main operation.
        result = self.inner_filter(inner_cube, original_standard_name)

        self._validate_inner_filter_result(result, cube)

        return result

    @abc.abstractmethod
    def inner_filter(self, cube, standard_name):
        """
        Perform the actual filtering calculation.

        This is the key operational method to be provided by every subclass, to
        implement its specific calculation.
        It is called by :meth:`~ugants.filter.generic.UnstructuredFilterABC.__call__`.

        Parameters
        ----------
        cube : :class:`~iris.cube.Cube`
            an input cube, on the filter mesh + location.  Its standard-name has
            been removed + is passed separately.

        standard_name: str or None
            The original input ``cube.standard_name``.  Removed in expectation that
            the result may not have the original phenomenon type.

        Returns
        -------
        :class:`iris.cube.Cube`
            a cube on the original mesh, with all other dimensions and components
            derived from the input cube.

        Notes
        -----
        Regarding the subclass-provided ``inner_filter`` function :

        * it must return a cube, which may be just the input cube with the new data
          installed.

        * the calculation should be repeated over all the non-mesh dimensions.

        * the result must have the same shape, mesh and mesh-location as the original.

        * other metadata (e.g. cell-methods, units) may be modified as required.

        * the phenomenon type of the result (standard name and cell-methods) may be
          set, or left empty.

        """
        pass
