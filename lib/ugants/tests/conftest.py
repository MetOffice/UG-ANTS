# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of UG-ANTS and is released under the BSD 3-Clause license.
# See LICENSE.txt in the root of the repository for full licensing details.
"""Source for pytest fixtures."""

from pathlib import Path
from shutil import rmtree
from tempfile import NamedTemporaryFile

import pytest


@pytest.fixture(scope="session")
def temporary_filepaths_function(tmp_path_factory):
    """
    Define a fixture to generate unique testfile paths.

    The fixture arg value is a function :

        `filepath_func(prefix: str = None, suffix: str = None) -> Path`

    The paths are guaranteed unique, can be created with prefix and suffix controls,
    and all exist in a common temporary directory, which is destroyed when the Pytest
    test session terminates.
    This is *unlike* the standard behaviour of Pytest tmp_path, tmp_path_factory or
    tmpdir_dir, all of which create directories which will persist on disk.

    Notes
    -----
        This is provided as a fixture because it works with the Pytest facilities
        for temporary files and directories.  The fixture has a "session" scope, so
        that it is quite certain *when* the common directory gets deleted, since other
        scope styles may not behave quite as expected in that respect.

    """
    # Create a common root directory for all our temporary files.
    base_dirpath = tmp_path_factory.mktemp("ugants_base")

    try:

        def _inner_filepath_func(
            prefix: str | None = None, suffix: str | None = None
        ) -> Path:
            """
            Return a unique filepath, at which a file can be created.

            The filepath is in a common subdirectory of the Pytest per-session
            temporary directory.  There is no file at the path.

            Parameters
            ----------
            prefix, suffix
                Filename controls, passed directly to the
                :class:`tempfile.NamedTemporaryFile` constructor.

            Returns
            -------
                temp_file_path

            """
            with NamedTemporaryFile(
                dir=str(base_dirpath), prefix=prefix, suffix=suffix, delete=True
            ) as tempfile:
                filepath = Path(tempfile.name)

            # The file no longer exists, but the caller can re-use its name.
            return filepath

        # Return a *context-manager* to create filepaths with given properties, in the
        # common temporary directory.
        yield _inner_filepath_func

    finally:
        # Forcibly remove the temporary path afterwards, so as not to leave
        # created data in the filesystem (which Pytest does, by design).  Our
        # scope="session" means it gets removed just before Pytest exits.
        rmtree(str(base_dirpath))
