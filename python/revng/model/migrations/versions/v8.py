#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

from revng.model.migrations import MigrationBase


class Migration(MigrationBase):
    def migrate(self, model):
        for function in model.get("Functions", {}):
            if "StackFrameType" in function:
                function["StackFrame"] = {"Type": function["StackFrameType"]}
                del function["StackFrameType"]
