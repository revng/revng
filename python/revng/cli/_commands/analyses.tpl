# Analyses

{% import 'common.tpl' as common %}

{% for analysis in data.analyses %}

## <a id="/analyses/{{analysis.name}}"></a>{{analysis.name}}

**Description**: {{analysis.doc}}

{{ common.shortlist('Steps using this analysis:', analysis.steps) }}

**Arguments**:
{% for argument in analysis.arguments %}
- **{{argument.name}}**. A container containing the following kinds:
  {% for kind in argument.acceptable_kinds %}
    - {{kind | link}}
  {% endfor %}
{% endfor %}

{% endfor %}
