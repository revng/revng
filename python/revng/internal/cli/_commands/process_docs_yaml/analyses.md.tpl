{#-
This template file is distributed under the MIT License. See LICENSE.md for details.
The notice below applies to the generated files.
#}

# Analyses

{% import 'common.md.tpl' as common -%}

This page reports the list of all the analyses currently featured in rev.ng.

{%- for branch in data.Branches -%}
{% for step in branch.Steps -%}
{% if step.Analyses -%}
{% for analysis in step.Analyses %}
## <a id="/analyses/{{analysis.Name}}"></a>`{{analysis.Name}}` analysis

```{bash notest}
revng analyze {{analysis.Name}}
```

{{- common.maybe('Runs after', "/pipeline/steps/" + step.Name) -}}

{{analysis.Docs}}
{% endfor -%}
{%- endif -%}
{%- endfor -%}
{%- endfor -%}
