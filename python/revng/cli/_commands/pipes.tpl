# Pipes

{% import 'common.tpl' as common %}

{% for pipe in data.pipes %}
## <a id="/pipes/{{pipe.name}}"></a>{{pipe.name}}

**Description**: {{pipe.doc}}

{{ common.shortlist('Used in', pipe.steps) }}

{% endfor %}
