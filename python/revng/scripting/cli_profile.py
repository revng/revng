#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

from revng.scripting.projects import CLIProject


class CLIProfile:
    def __init__(self, revng_executable: str | None = None):
        self.revng_executable = revng_executable

    def get_project(self, resume_dir: str | None = None) -> CLIProject:
        return CLIProject(resume_dir, self.revng_executable)
