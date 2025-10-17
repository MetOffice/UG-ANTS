# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of UG-ANTS and is released under the BSD 3-Clause license.
# See LICENSE.txt in the root of the repository for full licensing details.
"""General processing utility routines."""


def _check_and_normalise_index_in_range(dimension_index, number_of_dimensions, name):
    """
    Check that a dimension index is in a given range, and convert negative values.

    Convert any -N to (ndims - N).
    Error if the resulting index is < 0 or >= ndim.

    Parameters
    ----------
    dimension_index : int
        index of an array dimension
    number_of_dimensions : int
        number of dimensions of parent object
    name : str
        original arg-name, for constructing error message only

    Returns
    -------
    int
        a valid dimension number, always between 0 and (ndims - 1)
    """
    result = dimension_index
    if result < 0:
        result += number_of_dimensions
    if result < 0 or result >= number_of_dimensions:
        message = (
            f"Value of '{name}' arg is out of valid range "
            f"-{number_of_dimensions} to {number_of_dimensions - 1} : "
            f"got {dimension_index}."
        )
        raise ValueError(message)
    return result


def _one_dimension_transpose_indices(number_of_dimensions, from_dim, to_dim):
    """
    Return suitable transposition indices to re-locate one dimension of an array-like.

    Parameters
    ----------
    number_of_dimensions : int
        the number of dimensions in the array-like to be transposed.  Also, the number
        of indices in the result.
    from_dim : int
        the dimension number to 'move'.  ``-1`` means the last
    to_dim : int
        new index of the moved dimension.  ``0`` means to first dim, ``-1`` to the last.

    Returns
    -------
    list of int
        The appropriate indices to pass into a 'transpose' call,
        e.g. :func:`numpy.transpose` or :meth:`iris.cube.Cube.transpose`.
        Contains the values 0 .. ('ndims' - 1), in some permutation.

    """
    if number_of_dimensions == 0:
        raise ValueError("Cannot transpose with 0 dimensions.")

    from_dim = _check_and_normalise_index_in_range(
        from_dim, number_of_dimensions, "from_dim"
    )
    to_dim = _check_and_normalise_index_in_range(to_dim, number_of_dimensions, "to_dim")
    dims = list(range(number_of_dimensions))
    dims.remove(from_dim)
    dims = dims[:to_dim] + [from_dim] + dims[to_dim:]
    return dims


def move_one_dimension(content, from_dim, to_dim):
    """
    Move one dimension of an array.

    Transpose the input, to shift the selected dimension from one place to another in
    the index order.  Negative indices are also supported.

    Parameters
    ----------
    content : :class:`numpy.ndarray` or :class:`iris.cube.Cube`
        an array-like (cube, numpy or Dask array) to be transposed
    from_dim : int
        the dimension number to 'move'.  ``-1`` means the last
    to_dim : int
        index of the 'moved' dimension within the output.
        ``0`` means to the first dimension, ``-1`` to the last.

    Returns
    -------
    :class:`numpy.ndarray` or :class:`iris.cube.Cube`
        the result of an appropriate ``content.transpose()`` operation.
    """
    ndims = content.ndim
    dims = _one_dimension_transpose_indices(ndims, from_dim, to_dim)
    result = content.transpose(dims)
    return result
