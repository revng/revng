# Artifacts

{% import 'common.tpl' as common %}

{% for step in data.pipeline.steps %}
{% if step.artifact %}
## <a id="/pipeline/steps/{{step.name}}"></a>Artifact for {{step.name}} step

**Description**: {{step.artifact.doc}}

**Step**: {{ ("/pipeline/steps/" + step.name) | link }}

**Container**: {{ step.artifact.container }}

**Kind**: {{ step.artifact.kind | link }}

**Name for individual target**: `{{ step.artifact.single_target_filename }}`

{% endif %}
{% endfor %}
