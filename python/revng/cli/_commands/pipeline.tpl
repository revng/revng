# Pipeline

{% import 'common.tpl' as common %}

{{ common.graph(data.pipeline.steps) }}

The pipeline declares the following containers:
{% for container_declaration in data.pipeline.container_declarations %}
- **{{container_declaration.name}}**: {{container_declaration.type}}
{% endfor %}

{% for step in data.pipeline.steps %}
## <a id="/pipeline/steps/{{step.name}}"></a>{{step.name}} step

**Description**: {{step.doc}}

{{ common.maybe('Parent step', step.parent) }}

{% if step.artifact %}
**Artifact**: /artifact/{{step.name}}
{% endif %}

{{ common.longlist('Pipes', step.pipes) }}

{{ common.longlist('Analyses', step.analyses) }}

{% endfor %}
