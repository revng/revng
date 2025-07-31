#
# This file is distributed under the MIT License. See LICENSE.md for details.
#
import importlib.util
import logging
import os
import sys

logger = logging.getLogger(__name__)


def import_pipebox(module_path: str, is_complete: bool) -> object:
    """Import a module from a path. This is used to import the pypebox file.
    Args:
        env (dict[str, str]): The environment variables.
        module_path (str): The path to the module.
        is_complete (bool): If True, raise an error if the module is not found.
    """
    # Absolute path to the module
    module_abspath: str = os.path.abspath(module_path)
    if not os.path.exists(module_abspath):
        # This is a small trick to allow generating the module auto-complete
        # without having a pypebox file
        if is_complete:
            return object()
        logger.error(
            (
                "Pipebox file `%s` does not exist. Either set it using the "
                "PIPEBOX env var, or pass the --pipebox option."
            ),
            module_abspath,
        )
        sys.exit(1)
    # We guess that the module name is the file name without the extension
    module_name: str = os.path.basename(module_abspath.rstrip(".py"))
    # Dynamic import of the pypebox module
    spec = importlib.util.spec_from_file_location(module_name, module_abspath)
    if spec is None:
        if is_complete:
            return object()
        logger.error("Could not load module `%s` from `%s`", module_name, module_abspath)
        sys.exit(1)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        if is_complete:
            return object()
        logger.error("Could not load module `%s` from `%s`", module_name, module_abspath)
        sys.exit(1)
    # Execute the module to load it
    spec.loader.exec_module(module)
    return module
