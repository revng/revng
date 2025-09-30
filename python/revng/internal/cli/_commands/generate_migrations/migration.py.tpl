#
# This file is distributed under the MIT License. See LICENSE.md for details.
#

from revng.model.migrations import MigrationBase


class Migration(MigrationBase):
    def migrate(self, model):
        {%- for comment in comments %}
        {{ comment }}
        {%- endfor %}
        pass

