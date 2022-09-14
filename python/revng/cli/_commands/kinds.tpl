# Kinds

{% import 'common.tpl' as common %}

{{ common.graph(data.kinds) }}

{% for kind in data.kinds %}

## <a id="/kinds/{{kind.name}}"></a>{{kind.name}}

**Description**: {{kind.doc}}

**Rank**: {{ kind.rank | link }}
{{ common.maybe('Parent kind', kind.parent) }}

{{ common.shortlist('Child kinds', kind.children) }}

{{ common.shortlist('Artifacts using this kind', kind.artifacts) }}

{{ common.shortlist('Analyses using this kind', kind.analyses) }}

{% endfor %}
