# Ranks

{% import 'common.tpl' as common %}

{{ common.graph(data.ranks) }}

{% for rank in data.ranks %}

## <a id="/ranks/{{rank.name}}"></a>{{rank.name}}

**Description**: {{rank.doc}}

{{ common.maybe('Parent rank', rank.parent) }}

{{ common.shortlist('Child ranks', rank.children) }}

{{ common.shortlist('Kinds using this rank', rank.kinds) }}

{% endfor %}
