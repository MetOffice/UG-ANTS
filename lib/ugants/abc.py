# (C) Crown Copyright, Met Office. All rights reserved.
#
# This file is part of UG-ANTS and is released under the BSD 3-Clause license.
# See LICENSE.txt in the root of the repository for full licensing details.
"""Abstract base classes for UG-ANTS."""
import abc
import argparse
import inspect
from collections.abc import Sequence
from typing import Literal, get_args, get_origin

import iris
import iris.cube
from iris.experimental.ugrid import Mesh

from ugants.io import load, save


class Application(abc.ABC):
    """Abstract base class for creating UG-ANTS applications.

    The :class:`Application` base class provides a common interface for writing
    applications for generating unstructured grid ancillaries, providing
    functionality common to most applications:

    * A standardised command line interface is automatically generated from the
      application signature
    * Load and save methods are provided by the base class. These are intended to
      provide a sensible default, but can be overridden in specific applications
      if necessary.

    The :meth:`__init__` and :meth:`run` methods must be provided when subclassing
    :class:`Application`. See the documentation for these methods for more detail.
    """

    results: iris.cube.CubeList = None
    """A :class:`~iris.cube.CubeList` containing the results from
    running the application. This attribute is set when the
    :meth:`run` method is called."""
    output: str = None
    """Filepath to which the :attr:`results` will be written upon calling
    :meth:`save`. If the application has been instantiated using the
    :meth:`from_command_line` method, then this attribute will be set to the
    ``output`` command line argument."""

    _loader = load.ugrid

    @abc.abstractmethod
    def __init__(self, *args, **kwargs):
        """Initialise an instance of the class.

        Subclasses of :class:`Application` must provide an ``__init__``
        method.

        The ``__init__`` must be appropriately `type hinted
        <https://peps.python.org/pep-0484/>`_ in order for the command line
        argument parser to be constructed correctly.

        For example::

            from iris.cube import CubeList

            class ApplicationClass:
                def __init__(self, source: CubeList, scheme: str, number: int = 0):
                    self.source = source
                    self.scheme = scheme
                    self.number = number

        See Also
        --------
        :meth:`from_command_line` : for more information about how the type hints are
          used to construct the command line argument parser.
        """
        pass

    @abc.abstractmethod
    def run(self):
        """Run the application.

        Subclasses of :class:`ugants.abc.Application` must provide a ``run`` method.

        This method should make use of the inputs specified in the :meth:`__init__`
        method, and assign the results :class:`~iris.cube.CubeList` to :attr:`results`.
        """
        pass

    def save(self):
        """Save results to UGrid netcdf.

        Saves the :attr:`results` to the file location specfied by
        :attr:`output`.

        * :attr:`results` should be set by running :meth:`run`.
        * :attr:`output` should have been set from the command line,
          if the application instance has been created using the
          :meth:`from_command_line` method, or by explicitly setting :attr:`output`
          on the application instance.

        Raises
        ------
        ValueError
            If no output file location is specified at :attr:`output`.
        ValueError
            If no results are specified at :attr:`results`.
        """
        if self.output is None:
            raise ValueError("No output file location has been set.")
        if self.results is None:
            raise ValueError("The application has not yet been run, results is None.")
        save.ugrid(self.results, self.output)

    @classmethod
    def from_command_line(cls, args: Sequence[str] | None = None):
        """Create an application instance from the command line.

        The parameter names and type hints from the ``__init__`` method are used to
        construct a command line argument parser. The command line arguments are then
        parsed and passed to the ``__init__`` method to create an instance of the class.
        An ``output`` command line argument is also added, which is mapped to the
        :attr:`output` attribute when instantiating an application via the
        :meth:`from_command_line` method.

        Returns
        -------
        :
          An instance of the class initialised with the command line arguments.


        Notes
        -----
        Some parameter type hints have special effects on how the parser is constructed
        and the class instantiated:

        ======================================  ========================================
        Type hint                               Interpretation
        ======================================  ========================================
        :class:`~iris.cube.CubeList`            Add a command line positional argument
                                                representing the path to the UGrid data
                                                to be loaded. The loaded data will be
                                                passed to the ``__init__`` method to
                                                create an instance of the class. For
                                                example, ``source_data: CubeList`` adds
                                                a positional command line argument
                                                ``source_data``.

        :class:`~iris.experimental.ugrid.Mesh`  Add command line arguments representing
                                                the path to the mesh file to be loaded
                                                and the name of the mesh to extract.
                                                The loaded mesh will be passed to the
                                                ``__init__`` method to create an
                                                instance of the class. For example,
                                                ``target_mesh: Mesh`` adds two command
                                                line arguments: ``--target-mesh``
                                                (the path to the mesh file)
                                                and ``--target-mesh-name`` (the name of
                                                the mesh to extract).

        :obj:`typing.Literal`                   Add a command line argument with a
                                                limited set of choices. For example,
                                                ``regrid_scheme: Literal["conservative",
                                                "bilinear"]`` adds a command line
                                                argument ``--regrid-scheme`` with
                                                choices ``{conservative, bilinear}``.

        :obj:`bool`                             Add a command line flag that stores True
                                                if provided, or False if not provided.
                                                For example, ``a_flag: bool`` adds a
                                                command line flag ``--a-flag`` which
                                                sets ``a_flag`` to True if provided, or
                                                False if not provided.

        Parameter with default argument         Add an optional command line argument
                                                with a default value. For example,
                                                ``number: int = 0`` adds an optional
                                                command line argument ``--number`` with
                                                default value ``0``.
        ======================================  ========================================


        """
        parsed_arguments = vars(cls._parser().parse_args(args=args))

        app_signature = inspect.signature(cls).parameters.copy()
        for parameter_name, parameter in app_signature.items():
            if parameter.annotation is Mesh:
                mesh_name = parsed_arguments.pop(f"{parameter_name}_name")
                mesh_path = parsed_arguments[parameter_name]
                parsed_arguments[parameter_name] = load.mesh(mesh_path, mesh_name)

            if (
                parameter.annotation is iris.cube.CubeList
                and parsed_arguments[parameter_name] is not None
            ):
                # Parsed argument is currently a filepath, provided on the command line
                # Need to load a CubeList from the filepath, to pass to the __init__
                filepath = parsed_arguments[parameter_name]
                parsed_arguments[parameter_name] = cls._loader(filepath)

        # output is also included in the command line arguments,
        # but is not needed for the __init__
        output = parsed_arguments.pop("output")
        instance = cls(**parsed_arguments)
        instance.output = output
        return instance

    @classmethod
    def _parser(cls):
        """Return the command line argument parser for the application.

        Calls :meth:`add_argument` on each parameter in the signature of the
        class :meth:`__init__` to construct the :class:`argparse.ArgumentParser`
        instance. A positional command line argument ``output`` is also added.

        Returns
        -------
        parser : argparse.ArgumentParser
            A command line parser created from the signature of the class.
        """
        parser = argparse.ArgumentParser(
            description=inspect.getdoc(cls),
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )
        parser.allow_abbrev = False
        app_signature = inspect.signature(cls).parameters.copy()
        for parameter in app_signature.values():
            cls._add_argument(parser, parameter)
        parser.add_argument(
            "output", help=f"Path to output the results from {cls.__name__}."
        )
        return parser

    @staticmethod
    def _add_argument(parser: argparse.ArgumentParser, parameter: inspect.Parameter):
        """Add an argument to a command line parser from a function parameter.

        The provided ``parser`` is modified in place.

        A function parameter is mapped to command line arguments according to its
        annotation (type hint) and whether it is optional or required.

        The following logic is used to map parameter annotations to command line
        arguments:

        If annotation is iris.cube.CubeList:
          Add a positional command line argument with the same name as the parameter.
          The help string is 'Path to {parameter_name}'.
          This command line argument represents a file path and is used by the
          :meth:`from_command_line` method to load the data as a CubeList.

        All other annotations are "optional" command line arguments, e.g. ``--arg``.
        Note that this does not necessarily mean they are not required arguments,
        it just means they are not positional. The optional command line argument names
        are formatted from the parameter name by replacing all underscores with hyphens
        and adding the ``--`` prefix, e.g. the parameter name ``target_mesh`` becomes
        the command line option ``--target-mesh``.

        If annotation is iris.experimental.ugrid.Mesh:
          Add two required command line options: one representing the path to the mesh
          file, and one representing the name of the mesh to extract. The "mesh name"
          option is formatted by adding the ``-name`` suffix to the mesh path option.
          For example, for the parameter name ``target_mesh``, the two command line
          options are ``--target-mesh`` and ``--target-mesh-name``, representing the
          path to the mesh file and the name of the mesh to extract, respectively.

        If annotation is typing.Literal:
          Add a command line option with a restricted set of choices. See
          https://docs.python.org/3/library/typing.html#typing.Literal  and
          https://docs.python.org/3/library/argparse.html#choices for more details.
          For example, the function parameter ``scheme: typing.Literal["foo", "bar"]``
          will be mapped to the command line option ``--scheme`` with choices
          ``{"foo". "bar"}``.

        If the parameter has a default value:
          Add an optional, not required command line argument. For example, the function
          parameter ``number: int = 0`` is mapped to an optional command line argument
          ``--number``. If ``--number`` is not provided on the command line, the default
          value (in this case 0) will be passed.
          Note that the default value behaviour takes precedence over the CubeList
          behaviour, i.e. if a CubeList parameter has a default value, then it will
          become an optional command line argument, rather than a positional one.
          For example, the function parameter ``target_mask: CubeList = None`` is
          mapped to the optional, non-required command line argument ``--target-mask``.
          This is different from the usual behaviour for CubeList parameters, i.e.
          if the parameter has no default value ``target_mask: CubeList``, then
          the command line argument would be required positional ``target_mask``.

        Parameters
        ----------
        parser : argparse.ArgumentParser
            The parser to which to add the argument
        parameter : inspect.Parameter
            The function parameter from which to construct the command line argument
        """
        positional_name = parameter.name
        hyphened_name = positional_name.replace("_", "-")
        optional_name = f"--{hyphened_name}"
        # Construct dictionary of keyword arguments that will be
        # passed to parser.add_argument()
        add_argument_kwargs = {
            "type": parameter.annotation,
            "help": "",
        }
        if parameter.annotation is iris.cube.CubeList:
            name = positional_name
            add_argument_kwargs["help"] += f"Path to {name}."
            add_argument_kwargs["type"] = str
        else:
            # All non-CubeList parameters are optional parameters
            name = optional_name
            add_argument_kwargs["required"] = True

        # Handle a Mesh parameter.
        # We need to add a --mesh-name command line argument in order to load a mesh.
        if parameter.annotation is Mesh:
            add_argument_kwargs["help"] += f"Path to {positional_name}."
            add_argument_kwargs["type"] = str
            parser.add_argument(
                f"--{hyphened_name}-name",
                required=False,
                help="The name of the mesh contained in the "
                f"file provided to '{optional_name}'.",
                type=str,
            )

        # Parameters which take a restricted set of values are type-hinted
        # as typing.Literal[choice_1, choice_2, ...].
        # typing.get_args() returns these choices as a tuple.
        # Set "type" keyword argument to None, otherwise argparse will try
        # to call Literal on the parsed argument, which will fail.
        if get_origin(parameter.annotation) is Literal:
            add_argument_kwargs["choices"] = get_args(parameter.annotation)
            add_argument_kwargs["type"] = None

        if parameter.annotation is bool:
            add_argument_kwargs["action"] = "store_true"
            add_argument_kwargs.pop("type")
            add_argument_kwargs["required"] = False
            add_argument_kwargs["help"] = (
                f"If provided, ``{positional_name}`` will be True, otherwise False. "
                "See the API documentation for this application for more details."
            )

        # Handle default arguments.
        # Update "help" to show the default value in the CLI.
        # Default arguments should *always* be optional, even if they are
        # of type CubeList.
        if parameter.default is not parameter.empty:
            add_argument_kwargs["default"] = parameter.default
            add_argument_kwargs["help"] += f"Default is {parameter.default}."
            add_argument_kwargs["required"] = False
            name = optional_name

        parser.add_argument(name, **add_argument_kwargs)
