{#-
This template file is distributed under the MIT License. See LICENSE.md for details.
The notice below applies to the generated files.
#}

# Analyses

{% import 'common.md.tpl' as common -%}

This page reports the list of all the analyses currently featured in rev.ng.
{% for branch_name, branch in data.branches.items() -%}
{%- for task in branch.tasks -%}
{%- for analysis in task.analyses or [] %}
## <a id="/analyses/{{analysis.analysis}}"></a>`{{analysis.analysis}}` analysis

```{bash notest}
revng2 project analyze {{analysis.analysis}}
```

{{ common.maybe('Runs after', "/pipeline/branches/" + branch_name) -}}
{% if analysis.description %}{{ analysis.description }}{% endif %}
{% endfor -%}
{%- endfor -%}
{%- endfor -%}
{%- for analysis in data.analyses %}
{%- set name = analysis if analysis is string else analysis.analysis %}
## <a id="/analyses/{{name}}"></a>`{{name}}` analysis

```{bash notest}
revng2 project analyze {{name}}
```

{% if analysis is not string and analysis.description %}{{ analysis.description }}{% endif %}
{% endfor -%}
