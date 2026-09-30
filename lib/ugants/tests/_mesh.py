# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of UG-ANTS and is released under the BSD 3-Clause license.
# See LICENSE.txt in the root of the repository for full licensing details.
# Some of the content of this file has been produced with the assistance of
# Met Office GitHub Copilot Enterprise.
import numpy as np
import pyvista as pv
from iris.coords import AuxCoord
from iris.experimental.ugrid import Connectivity, Mesh


def _polydata_to_mesh(polydata: pv.PolyData):
    cell_centres_polydata = polydata.cell_centers()

    node_lats, node_lons = calculate_lat_lon(polydata.points)
    face_lats, face_lons = calculate_lat_lon(cell_centres_polydata.points)

    face_node_indices = np.ma.asarray(polydata.regular_faces)

    n_faces = polydata.n_cells
    face_face_indices = np.ma.masked_all((n_faces, 4), dtype=int)
    for face_id in range(n_faces):
        connected_faces = polydata.cell_neighbors(face_id, "edges")
        face_face_indices[face_id, : len(connected_faces)] = connected_faces

    node_x_aux = AuxCoord(
        points=node_lons,
        standard_name="longitude",
        units="degrees_east",
        long_name="node_x_coordinates",
    )
    node_y_aux = AuxCoord(
        points=node_lats,
        standard_name="latitude",
        units="degrees_north",
        long_name="node_y_coordinates",
    )
    face_x_aux = AuxCoord(
        points=face_lons,
        standard_name="longitude",
        units="degrees_east",
        long_name="face_x_coordinates",
    )
    face_y_aux = AuxCoord(
        points=face_lats,
        standard_name="latitude",
        units="degrees_north",
        long_name="face_y_coordinates",
    )
    face_node_connectivity = Connectivity(
        indices=face_node_indices, cf_role="face_node_connectivity", start_index=0
    )
    face_face_connectivity = Connectivity(
        indices=face_face_indices, cf_role="face_face_connectivity", start_index=0
    )
    mesh = Mesh(
        long_name="my_mesh",
        topology_dimension=2,
        node_coords_and_axes=[(node_x_aux, "x"), (node_y_aux, "y")],
        connectivities=[face_node_connectivity, face_face_connectivity],
        face_coords_and_axes=[(face_x_aux, "x"), (face_y_aux, "y")],
    )
    return mesh


def calculate_lat_lon(points):
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    xy = np.hypot(x, y)
    lats = np.degrees(np.arctan2(z, xy))
    lons = np.degrees(np.arctan2(y, x))
    return lats, lons
