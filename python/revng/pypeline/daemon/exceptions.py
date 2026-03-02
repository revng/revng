#
# This file is distributed under the MIT License. See LICENSE.md for details.
#


class DaemonException(Exception):
    """
    An exception that a route can raise to do an early-exit with
    some data and a status code.
    """

    def __init__(self, code: int, body: dict):
        self.code = code
        self.body = {"type": self.__class__.__name__, **body}


class EpochError(DaemonException):
    def __init__(self, current_epoch: int, provided_epoch: int):
        super().__init__(
            409,
            {
                "expected_epoch": current_epoch,
                "message": (
                    f"Epoch mismatch: client has epoch {provided_epoch}, "
                    f"server has epoch {current_epoch}."
                ),
            },
        )


class MalformedRequestError(DaemonException):
    def __init__(self, message: str):
        super().__init__(400, {"message": message})
