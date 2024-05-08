#
# This file is distributed under the MIT License. See LICENSE.mit for details.
#


class RevngException(Exception):
    pass


class RevngDocumentException(RevngException):
    def __init__(self, document_error):
        self.document_error = document_error
        error_text = "Document Error with the following reasons:\n"
        for reason in document_error.reasons:
            error_text += f"* {reason.message} at {reason.location}\n"
        super().__init__(error_text)


class RevngSimpleException(RevngException):
    def __init__(self, simple_error):
        self.simple_error = simple_error
        super().__init__(simple_error.message)
