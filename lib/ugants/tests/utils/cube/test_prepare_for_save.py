# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of UG-ANTS and is released under the BSD 3-Clause license.
# See LICENSE.txt in the root of the repository for full licensing details.
"""Tests for :func:`ugants.utils.cube.prepare_for_save`."""
import datetime
from unittest.mock import Mock, patch

import pytest
from iris.cube import Cube, CubeList
from ugants.exceptions import ProvisionalWarning
from ugants.utils.cube import prepare_for_save

PROVISIONAL_WARNING_MESSAGE = (
    "The results output by UG-ANTS are provisional and subject to change."
)
# Ignore the provisional output warning message for all tests
pytestmark = pytest.mark.filterwarnings(
    f"ignore:{PROVISIONAL_WARNING_MESSAGE}:ugants.io.save.ProvisionalWarning"
)


@pytest.fixture(autouse=True)
def _fix_time_and_commandargs():
    """Fix the time and command-line args, as seen by the tested function.

    Use Mock.patch to set timestamp and command-line args to plausible values.
    """
    # Some general fake context seen by the utils module within these tests.
    # A dummy 'sys.argv'
    dummy_commandline_args = [
        "/path/to/testapp",
        "-k",
        "this",
        "--k2",
        "that",
        "arg1",
        "arg2",
    ]
    # A dummy 'datetime.today()'
    dummy_datetime = datetime.datetime(2001, 2, 3, 14, 55, 26)
    mock_datetime_today_call = Mock(return_value=dummy_datetime)
    mock_datetime = Mock(today=mock_datetime_today_call)
    with patch("ugants.utils.cube.sys.argv", dummy_commandline_args):
        with patch("ugants.utils.cube.datetime", mock_datetime):
            yield


def _make_testcube(name=None, initial_history=None):
    """Produce a suitable basic test-cube with a known history attribute."""
    if name is None:
        name = "test-cube"
    cube = Cube([0])
    cube.rename(name)
    if initial_history is not None:
        cube.attributes["history"] = initial_history
    return cube


class TestSingleCube:
    # Expected history entry
    expected_entry_text = "2001-02-03T14:55:26: testapp -k this --k2 that arg1 arg2"
    # Default values for the basic default test-cube, and expected result.
    initial_history_text = "Initial-content."

    def test_cube_with_no_initial_history(self):
        # Check with an initially empty, or missing, history attribute
        # N.B. notably, there should be no extra "\n" at the end.
        test_cube = _make_testcube(initial_history=None)

        updated_cube = prepare_for_save(test_cube)

        # Result is the default "entry" text alone : no additional newline.
        expected = self.expected_entry_text
        assert updated_cube.attributes["history"] == expected

    def test_cube_with_initial_history(self):
        # Check basic operation
        test_cube = _make_testcube(initial_history=self.initial_history_text)

        updated_cube = prepare_for_save(test_cube)

        expected_history = f"{self.expected_entry_text}\n{self.initial_history_text}"
        assert updated_cube.attributes["history"] == expected_history

    def test_operation_leaves_original_cube_unaltered(self):
        # Check cube is not modified in place
        test_cube = _make_testcube()
        original_test_cube = test_cube.copy()
        prepare_for_save(test_cube)
        assert test_cube == original_test_cube

    def test_empty_string_initial_history(self):
        # When the existing is an empty string, it adds a newline separator, just
        # as for a non-empty string.
        test_cube = _make_testcube(initial_history="")

        updated_cube = prepare_for_save(test_cube)

        # Result is the default "entry" text, PLUS a newline.
        expected = self.expected_entry_text + "\n"
        assert updated_cube.attributes["history"] == expected

    def test_preserves_newlines(self):
        initial_text = "\n\nEntry 1\n\nEntry 2\n\n"
        test_cube = _make_testcube(initial_history=initial_text)

        updated_cube = prepare_for_save(test_cube)

        # Result preserves original content unchanged.
        expected = self.expected_entry_text + "\n" + initial_text
        assert updated_cube.attributes["history"] == expected

    def test_preserves_spaces(self):
        initial_text = "one \n  two.  \n \nThree."  # N.B. one entry is blank
        test_cube = _make_testcube(initial_history=initial_text)

        updated_cube = prepare_for_save(test_cube)

        # Result preserves original content unchanged.
        expected = self.expected_entry_text + "\n" + initial_text
        assert updated_cube.attributes["history"] == expected

    def test_nonstring_initial_history__fail(self):
        # A non-string initial history attribute causes an error.
        test_cube = _make_testcube(initial_history=3.0)

        expected_message = (
            r"""TypeError("Cube 'test-cube' has a 'history' attribute """
            'of non-string type : 3.0.")'
        )
        with pytest.raises(Exception) as error:
            prepare_for_save(test_cube)

        assert repr(error.value) == expected_message

    def test_ugants_status(self):
        test_cube = _make_testcube()
        assert "ugants_status" not in test_cube.attributes
        updated_cube = prepare_for_save(test_cube)
        assert updated_cube.attributes["ugants_status"] == PROVISIONAL_WARNING_MESSAGE

    def test_suite_provenance_set(self, monkeypatch):
        test_cube = _make_testcube()
        assert "suite_provenance" not in test_cube.attributes
        monkeypatch.setenv("SUITE_PROVENANCE", "r1234_branch@5678")
        updated_cube = prepare_for_save(test_cube)
        assert updated_cube.attributes["suite_provenance"] == "r1234_branch@5678"

    def test_suite_provenance_none(self, monkeypatch):
        test_cube = _make_testcube()
        assert "suite_provenance" not in test_cube.attributes
        monkeypatch.delenv("SUITE_PROVENANCE", raising=False)
        updated_cube = prepare_for_save(test_cube)
        assert updated_cube.attributes["suite_provenance"] == "None"

    def test_provisional_warning_raised(self):
        test_cube = _make_testcube()
        with pytest.warns(ProvisionalWarning, match=PROVISIONAL_WARNING_MESSAGE):
            prepare_for_save(test_cube)


class TestCubeList:
    # Expected history entry
    expected_entry_text = "2001-02-03T14:55:26: testapp -k this --k2 that arg1 arg2"
    # Default values for the basic default test-cube, and expected result.
    initial_history_text = "Initial-content."

    def test_single_cube_cubelist(self):
        # Check operation with multiple cubes.
        test_cube = _make_testcube(initial_history=self.initial_history_text)
        cubes = CubeList([test_cube])

        updated_cubes = prepare_for_save(cubes)

        assert isinstance(updated_cubes, CubeList)
        assert len(updated_cubes) == 1

        expected = f"{self.expected_entry_text}\n{self.initial_history_text}"
        assert updated_cubes[0].attributes["history"] == expected

    def test_multiple_cubes_same_history(self):
        # Check operation with multiple cubes with the same initial history.
        test_cube_1 = _make_testcube(initial_history=self.initial_history_text)
        test_cube_2 = _make_testcube(
            name="test_cube_2", initial_history=self.initial_history_text
        )
        cubes = CubeList([test_cube_1, test_cube_2])

        updated_cubes = prepare_for_save(cubes)

        assert isinstance(updated_cubes, CubeList)
        assert len(updated_cubes) == 2

        # Both get the expected 'standard' result.
        expected = f"{self.expected_entry_text}\n{self.initial_history_text}"
        assert updated_cubes[0].attributes["history"] == expected
        assert updated_cubes[1].attributes["history"] == expected

    def test_multiple_cubes_different_histories(self):
        # Check operation with multiple cubes with different initial histories.
        test_cube_1 = _make_testcube(initial_history=self.initial_history_text)
        test_cube_2 = _make_testcube(
            name="test_cube_2", initial_history="Cube 2 history."
        )
        cubes = CubeList([test_cube_1, test_cube_2])

        updated_cubes = prepare_for_save(cubes)

        assert isinstance(updated_cubes, CubeList)
        assert len(updated_cubes) == 2

        # Both get the expected 'standard' result.
        expected_history_1 = f"{self.expected_entry_text}\n{self.initial_history_text}"
        expected_history_2 = f"{self.expected_entry_text}\nCube 2 history."
        assert updated_cubes[0].attributes["history"] == expected_history_1
        assert updated_cubes[1].attributes["history"] == expected_history_2

    def test_ugants_status(self):
        test_cube_1, test_cube_2 = _make_testcube(), _make_testcube()
        test_cubelist = CubeList([test_cube_1, test_cube_2])
        for test_cube in test_cubelist:
            assert "ugants_status" not in test_cube.attributes

        updated_cubelist = prepare_for_save(test_cubelist)
        for updated_cube in updated_cubelist:
            assert (
                updated_cube.attributes["ugants_status"] == PROVISIONAL_WARNING_MESSAGE
            )

    def test_suite_provenance_set(self, monkeypatch):
        monkeypatch.setenv("SUITE_PROVENANCE", "r1234_branch@5678")
        test_cube_1, test_cube_2 = _make_testcube(), _make_testcube()
        test_cubelist = CubeList([test_cube_1, test_cube_2])
        for test_cube in test_cubelist:
            assert "suite_provenance" not in test_cube.attributes

        updated_cubelist = prepare_for_save(test_cubelist)
        for updated_cube in updated_cubelist:
            assert updated_cube.attributes["suite_provenance"] == "r1234_branch@5678"

    def test_suite_provenance_none(self, monkeypatch):
        monkeypatch.delenv("SUITE_PROVENANCE", raising=False)
        test_cube_1, test_cube_2 = _make_testcube(), _make_testcube()
        test_cubelist = CubeList([test_cube_1, test_cube_2])
        for test_cube in test_cubelist:
            assert "suite_provenance" not in test_cube.attributes

        updated_cubelist = prepare_for_save(test_cubelist)
        for updated_cube in updated_cubelist:
            assert updated_cube.attributes["suite_provenance"] == "None"

    def test_provisional_warning_raised(self):
        test_cube_1, test_cube_2 = _make_testcube(), _make_testcube()
        test_cubelist = CubeList([test_cube_1, test_cube_2])
        with pytest.warns(ProvisionalWarning, match=PROVISIONAL_WARNING_MESSAGE):
            prepare_for_save(test_cubelist)
