#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

"""
CABC: Compatibility ABCs.

This is needed to support proper type checking, but
avoid metaclasses which we should avoid on classes we intend of implementing in
C++.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from abc import ABC, abstractmethod
else:

    class ABC:
        pass

    def abstractmethod(func):
        # Import it inside so users cannot import it by mistake
        from functools import wraps

        @wraps(func)
        def wrapper(*args, **kwargs):
            raise NotImplementedError("Abstract method")

        return wrapper
