# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of UG-ANTS and is released under the BSD 3-Clause license.
# See LICENSE.txt in the root of the repository for full licensing details.
"""IO themed UG-ANTS applications."""

from collections.abc import Sequence
from dataclasses import dataclass

from iris.experimental.ugrid import Mesh

from ugants.abc import Application
from ugants.io import save


@dataclass
class ExtractSingleMesh(Application):
    """Extract a single named mesh from a UGrid netCDF file.

    This application should only be run from the command line, not instantiated
    directly in code. The mesh provided to the ``__init__`` is expected to have
    already been extracted from file in the :meth:`from_command_line` method.

    This application has no ``run`` method, since there is no processing required.
    The application just calls :func:`ugants.io.save.mesh` on the loaded mesh.

    Example
    -------
    >>> extract_mesh_app = ExtractSingleMesh.from_command_line()
    >>> extract_mesh_app.save()
    """

    mesh: Mesh
    """The single mesh extracted from file."""

    def run(self):
        """Not implemented. Only exists to meet the Application ABC requirements.

        :meta private:
        """
        raise RuntimeError("The run method is not implemented for this application.")

    @classmethod
    def from_command_line(cls, args: Sequence[str] | None = None):
        """Create an application instance from the command line.

        The following arguments are parsed:

        * ``--mesh``: The path to the mesh file from which to extract a mesh.
        * ``--mesh-name``: The ``var_name`` of the mesh to be extracted.
        * ``output``: The output file location to which to save the extracted mesh.

        An instance of the :class:`ExtractSingleMesh` class is created, instantiated
        with the mesh extracted from the provided file.

        Parameters
        ----------
        args : ~collections.abc.Sequence[str] | None
            Command line arguments to parse, by default None. If None, then the
            arguments are parsed from the command line.

        Returns
        -------
        :class:`ExtractSingleMesh`
            An instance of the class.
        """
        return super().from_command_line(args)

    def save(self):
        """Save the extracted mesh to UGrid netCDF.

        Calls :func:`ugants.io.save.mesh` on the extracted mesh.
        """
        save.mesh(self.mesh, self.output)
