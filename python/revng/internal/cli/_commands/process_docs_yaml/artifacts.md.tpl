{#-
This template file is distributed under the MIT License. See LICENSE.md for details.
The notice below applies to the generated files.
#}

# Artifacts

{% import 'common.md.tpl' as common -%}

This page reports the list of all the artifacts that can be produced by rev.ng.

{%- for branch in data.Branches -%}
{% for step in branch.Steps -%}
{%- if step.Artifacts %}
## <a id="/artifacts/{{step.Name}}"></a>`{{step.Name}}` artifact

```{bash notest}
revng artifact {{step.Name}}
```

{{- common.maybe('Step', "/pipeline/steps/" + step.Name) -}}
**File name**: `{{ step.Artifacts.Container }}`

{{step.Artifacts.Docs}}
{% endif -%}
{%- endfor -%}
{%- endfor -%}
