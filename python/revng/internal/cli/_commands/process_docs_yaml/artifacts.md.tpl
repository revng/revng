{#-
This template file is distributed under the MIT License. See LICENSE.md for details.
The notice below applies to the generated files.
#}

# Artifacts

{% import 'common.md.tpl' as common -%}

This page reports the list of all the artifacts that can be produced by rev.ng.
{% for branch_name, branch in data.branches.items() -%}
{%- for task in branch.tasks -%}
{%- for artifact in task.artifacts or [] %}
## <a id="/artifacts/{{artifact.name}}"></a>`{{artifact.name}}` artifact

```{bash notest}
revng2 project artifact {{artifact.name}}
```

{{ common.maybe('Branch', "/pipeline/branches/" + branch_name) -}}
{%- if artifact.filename %}
**File name**: `{{ artifact.filename }}`
{% endif %}
{% if artifact.description %}{{ artifact.description }}{% endif %}
{% endfor -%}
{%- endfor -%}
{%- endfor -%}
