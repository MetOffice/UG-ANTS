Writing an ancillary application with UG-ANTS
=============================================

The :class:`ugants.abc.Application` abstract base class provides provides a
common interface for writing applications for generating unstructured grid
ancillaries, providing functionality common to most applications:

* A standardised command line interface is automatically generated from the
  application signature.
* Load and save methods are provided by the base class. These are intended to
  provide a sensible default, but can be overridden in specific applications
  if necessary.

The :meth:`~ugants.abc.Application.__init__` and :meth:`~ugants.abc.Application.run`
methods must be provided when subclassing :class:`~ugants.abc.Application`.
See the documentation for these methods for more detail.

Example user-defined application
--------------------------------
A user defined application is created by subclassing the
:class:`~ugants.abc.Application` base class, and providing the required
:meth:`~ugants.abc.Application.__init__` and :meth:`~ugants.abc.Application.run` methods:

.. code-block:: python

    # example_application.py
    from iris.cube import CubeList
    from iris.experimental.ugrid import Mesh
    from typing import Literal

    class UserDefinedApplication(Application):
        '''An example user-defined application inheriting from the Application ABC.

        Parameters
        ----------
        source
            UGrid source data.
        target_mesh
            The target mesh to regrid to.
        number
            A number.
        scheme
            The regrid scheme to be used.
        option
            An arbitrary option.
        '''
        def __init__(
            self,
            source: CubeList,
            target_mesh: Mesh,
            number: int,
            scheme: Literal["conservative", "bilinear"] = "conservative",
            option: str = None
        ):
            self.source = source
            self.target_mesh = target_mesh
            self.number = number
            self.scheme = scheme
            self.option = option

        def run(self):
            '''Run the UserDefinedApplication.

            In this case, just print some of the application attributes,
            then set self.result.
            '''
            print(f"Running application")
            print(f"Using scheme '{self.scheme}' and number {self.number}")
            print(f"Option is: {self.option}")
            self.results = self.source

    if __name__ == "__main__":
        app_instance = UserDefinedApplication.from_command_line()
        app_instance.run()
        app_instance.save()

The most common way of instantiating and running an application is via the command
line interface. This is achieved by calling the :meth:`~ugants.abc.Application.from_command_line`
method, which will parse arguments from the command line. In the case of the
``UserDefinedApplication`` example above, the following command line interface will
be generated:

.. code-block::

    usage: example_application.py [-h] --target-mesh-name TARGET_MESH_NAME
                                    --target-mesh TARGET_MESH --number NUMBER
                                    [--scheme {conservative,bilinear}]
                                    [--option OPTION]
                                    source output

    An example user-defined application inheriting from the Application ABC.

    Parameters
    ----------
    source
        UGrid source data.
    target_mesh
        The target mesh to regrid to.
    number
        A number.
    scheme
        The regrid scheme to be used.
    option
        An arbitrary option.

    positional arguments:
    source                Path to source.
    output                Path to output the results from
                            UserDefinedApplication.

    options:
    -h, --help            show this help message and exit
    --target-mesh-name TARGET_MESH_NAME
                            The name of the mesh contained in the file provided to
                            '--target-mesh'.
    --target-mesh TARGET_MESH
                            Path to target_mesh.
    --number NUMBER
    --scheme {conservative,bilinear}
                            Default is conservative.
    --option OPTION       Default is None.

The application is then run and the results saved by calling the :meth:`~ugants.abc.Application.run` and
:meth:`~ugants.abc.Application.save` methods. The results will be saved to the file specified by the
:attr:`~ugants.abc.Application.output` attribute on the class instance, which is automatically set by the
``output`` command line argument.

The above python script is then run as follows:

.. code-block:: console

    $ python example_application.py --target-mesh-name dynamics --target-mesh path/to/target/mesh.nc --number 1 --scheme conservative --option foo path/to/source.nc path/to/output.nc
    Running application
    Using scheme 'conservative' and number 1
    Option is: foo

An application can be also instantiated and run using the API. Using the above
application class, and assuming a :class:`~iris.cube.CubeList` ``source`` and a
:class:`~iris.experimental.ugrid.Mesh` ``mesh`` have already been loaded, the
application is instantiated and run as follows:

.. code-block:: python

    >>> app_instance = UserDefinedApplication(source=source, target_mesh=mesh, number=3, scheme="bilinear")
    >>> app_instance.run()
    Running application
    Using scheme 'bilinear' and number 3
    Option is: None

The results are stored in the :attr:`~ugants.abc.Application.results` attribute. To save the results, set
the :attr:`~ugants.abc.Application.output` attribute, then call :meth:`~ugants.abc.Application.save`.

.. code-block:: python

    >>> app_instance.output = "path/to/output/result.nc"
    >>> app_instance.save()
