{#-
This template file is distributed under the MIT License. See LICENSE.md for details.
The notice below applies to the generated files.
#}

# Pipeline

{% import 'common.md.tpl' as common -%}

## Overview

The rev.ng pipeline is organized into *branches*.
Each branch is a sequence of *tasks*.
A branch starts from the end of another branch (its parent), so the branches
form a tree; the place where a branch departs from its parent is a *branching
point*.

Each task is either:

- a *pipe*, which transforms the contents of one or more *containers*;
- a *savepoint*, which caches the contents of some containers so that later runs
  can resume the pipeline from there.

A task can also expose an *artifact* (an output the user can request) or run an
*analysis* (which refines the [model](../user-manual/key-concepts/model.md)).

The following graph shows the branches, their branching points, and the
artifacts and analyses attached to each of them. It uses this notation:

- a small empty circle is a *branch point*;
- a rounded box is an *artifact*;
- a folder is an *artifact produced at a savepoint*;
- a dashed box is an *analysis*.

```dot
digraph {
  bgcolor = transparent;
  node [shape=box,color="#6c7278",fontname="monospace",fillcolor="#24282f",fontcolor=white,style="filled",width="1.3"];
  edge [color=white,fontcolor=white,fontname=monospace];
  rankdir = TB;
  ratio = compress;
  size = "20";

{#- One node per branch: a small empty circle; this is the branching backbone. #}
{%- for branch_name, branch in data.branches.items() %}
  "branch-{{branch_name}}" [label="",shape=circle,fixedsize=true,width="0.2",height="0.2",style=solid,URL="#/pipeline/branches/{{branch_name}}"];
{%- endfor %}

{#- Artifact and analysis nodes. Savepoints are not shown as nodes, but an
    artifact produced at a savepoint is drawn with the folder shape. #}
{%- for branch_name, branch in data.branches.items() %}
{%- for task in branch.tasks %}
{%- for artifact in task.artifacts or [] %}
{%- if task.savepoint %}
  "artifact-{{artifact.name}}" [label="{{artifact.name | replace('-', '-\\n')}}",URL="../artifacts/#/artifacts/{{artifact.name}}",shape=folder];
{%- else %}
  "artifact-{{artifact.name}}" [label="{{artifact.name | replace('-', '-\\n')}}",URL="../artifacts/#/artifacts/{{artifact.name}}",style="rounded,filled"];
{%- endif %}
{%- endfor %}
{%- for analysis in task.analyses or [] %}
  "analysis-{{analysis.analysis}}" [label="{{analysis.analysis | replace('-', '-\\n')}}",URL="../analyses/#/analyses/{{analysis.analysis}}",style="filled,dashed"];
{%- endfor %}
{%- endfor %}
{%- endfor %}

{#- Solid flow: each branch runs from its parent's branch point, through the
    branch's artifacts (in task order), to the branch's own branch point. This
    places every artifact inline, between the two branch points it sits between. #}
{% for branch_name, branch in data.branches.items() -%}
{%- set ns = namespace(prev = ("branch-" + branch.from) if branch.from else none) %}
{%- for task in branch.tasks %}
{%- for artifact in task.artifacts or [] %}
{%- if ns.prev is not none %}
{{ emit_edge(ns.prev, "artifact-" + artifact.name) }}
{%- endif %}
{%- set ns.prev = "artifact-" + artifact.name %}
{%- endfor %}
{%- endfor %}
{%- if ns.prev is not none %}
{{ emit_edge(ns.prev, "branch-" + branch_name) }}
{%- endif %}
{%- endfor %}

{#- Analyses hang off the branch's last artifact, or off the branch point itself
    when the branch produces no artifact. #}
{% for branch_name, branch in data.branches.items() -%}
{%- set ns = namespace(target = "branch-" + branch_name) %}
{%- for task in branch.tasks %}
{%- for artifact in task.artifacts or [] %}
{%- set ns.target = "artifact-" + artifact.name %}
{%- endfor %}
{%- endfor %}
{%- for task in branch.tasks %}
{%- for analysis in task.analyses or [] %}
  "{{ns.target}}" -> "analysis-{{analysis.analysis}}" [style=dashed,arrowhead=none,color="#6c7278"];
{%- endfor %}
{%- endfor %}
{%- endfor %}
}
```

## Containers

The pipeline declares the following containers:
{% for container in data.containers %}
- `{{container.name}}` (type: `{{container.type}}`)
{%- endfor %}

## Branches

{% for branch_name, branch in data.branches.items() %}
### <a id="/pipeline/branches/{{branch_name}}"></a>`{{branch_name}}` branch
{% if branch.from %}
**Starts from**: {{ ("/pipeline/branches/" + branch.from) | link }}
{% endif %}
**Tasks**:
{% for task in branch.tasks %}
{%- if task.pipe %}
- `{{task.pipe}}({{ (task.arguments or []) | join(", ") }})`
{%- elif task.savepoint %}
- `savepoint("{{task.savepoint}}", {% for c in task.containers %}{% if loop.index > 1 %}, {% endif %}{{c}}{% endfor %})`
{%- endif %}
{%- for artifact in task.artifacts or [] %}
    - Artifact: {{ ("/artifacts/" + artifact.name) | link }}
{%- endfor %}
{%- for analysis in task.analyses or [] %}
    - Analysis: {{ ("/analyses/" + analysis.analysis) | link }}
{%- endfor %}
{%- endfor %}
{% endfor %}

## Analysis lists

{% for list in data["analysis-lists"] -%}
- `{{list.name}}`
{% for analysis in list.analyses %}
    - {{("/analyses/" + analysis) | link}}
{%- endfor -%}
{%- endfor -%}
