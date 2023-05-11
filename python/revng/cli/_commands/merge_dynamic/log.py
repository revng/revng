#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

import sys

verbose = True


def set_verbose(verb: bool):
    global verbose
    verbose = verb


def log(message):
    # TODO: implement properly using python logging and setting the loglevel
    if verbose:
        sys.stderr.write(message + "\n")
