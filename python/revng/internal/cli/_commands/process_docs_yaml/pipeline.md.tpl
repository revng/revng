{#-
This template file is distributed under the MIT License. See LICENSE.md for details.
The notice below applies to the generated files.
#}

# Pipeline

{% import 'common.md.tpl' as common -%}

{%- set parent_step = namespace(value="") -%}

## Overview

The rev.ng pipeline is composed by *steps*.
Each step runs a series of *pipes*.
Each pipe works on one or more *container*.
Certain steps have an *artifact*, which represents basically their output.

[Analyses](../user-manual/model.md), when scheduled, run at a certain point in the pipeline, after a specified step.

The following tree reports the structure of the pipeline, its steps and analyses.

```graphviz dot pipeline.svg
digraph {
  bgcolor = transparent;
  node [shape=box,color="#6c7278",fontname="monospace",fillcolor="#24282f",fontcolor=white,style="filled",width="1.3"];
  edge [color=white,fontcolor=white,fontname=monospace];
  rankdir = TB;


  legend [label="Legend:",style=none,color=transparent]
  "legend-step" [label="Step"]
  "legend-artifact" [label="Step with\nartifact",style="filled,rounded"]
  "legend-analysis" [label="Analysis",style="filled,dashed"]
  "legend" -> "step-initial" [color=transparent]

{%- for branch in data.Branches -%}
{% for step in branch.Steps %}
  "step-{{step.Name}}" [label="{{step.Name | replace('-', '-\\n')}}",URL="#/pipeline/steps/{{step.Name}}"{{ ",style=\"rounded,filled\"" if step.Artifacts}}];
{%- for analysis in step.Analyses %}
  "analysis-{{analysis.Name}}" [label="{{analysis.Name  | replace('-', '-\\n')}}"URL="#/pipeline/analyses/{{analysis.Name}}",style="filled,dashed"];
{%- endfor -%}
{%- endfor -%}
{%- endfor %}

{% set parent_step.value = "" -%}
{%- set successors = namespace(value={}) -%}

{%- for branch in data.Branches -%}

{%- if branch.From -%}
{%- set parent_step.value = branch.From -%}
{%- endif -%}
{% for step in branch.Steps %}
{%- if parent_step.value %}
{{ emit_edge("step-" + parent_step.value, "step-" + step.Name) }}
{%- endif -%}
{% for analysis in step.Analyses %}
{{ emit_edge("step-" + step.Name, "analysis-" + analysis.Name) }}
{%- endfor -%}
{%- set parent_step.value = step.Name -%}
{%- endfor -%}
{%- endfor %}
}
```

## Containers

The pipeline declares the following containers:
{% for container_declaration in data.Containers %}
- `{{container_declaration.Name}}` (type: `{{container_declaration.Type}}`)
{%- endfor %}

## Steps

{% set parent_step.value = "" -%}

{%- for branch in data.Branches -%}

{%- if branch.From -%}
{%- set parent_step.value = '/pipeline/steps/' + branch.From -%}
{% endif %}

{% for step in branch.Steps %}
### <a id="/pipeline/steps/{{step.Name}}"></a>`{{step.Name}}` step

{{- common.maybe('Parent step', parent_step.value) -}}

{%- if step.Artifacts -%}
**Artifact**: {{step.Artifacts.Docs}}
{%- endif %}

{% if step.Pipes -%}
**Pipes**:

{% for pipe in step.Pipes -%}
- `{{pipe.Type}}({{ pipe.UsedContainers | join(", ") }})`{% if pipe.Passes %}{% for pass in pipe.Passes %}
    - `{{pass}}`
{%- endfor %}{% endif %}
{% endfor -%}
{%- endif %}

{% if step.Analyses -%}
**Analyses**:

{% for analysis in step.Analyses %}
- <a id="/pipeline/analyses/{{analysis.Name}}"></a>`{{analysis.Name}}({{ analysis.UsedContainers | join(", ") }})`: {{analysis.Docs | indent(4)}}
{% endfor -%}
{%- endif -%}

{%- set parent_step.value = '/pipeline/steps/' + step.Name -%}

{%- endfor -%}
{%- endfor -%}


## Analysis lists

{% for list in data.AnalysesLists -%}
- `{{list.Name}}`
{% for analysis in list.Analyses %}
    - {{("/pipeline/analyses/" + analysis) | link}}
{%- endfor -%}
{%- endfor -%}
