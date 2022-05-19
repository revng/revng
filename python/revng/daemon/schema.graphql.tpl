{#
  This file is distributed under the MIT License. See LICENSE.md for details.
#}
type Query {
    info: Info!
    step(name: String!): Step
    container(name: String!, step: String!): Container
    targets(pathspec: String!): [Step!]!
    produce(step: String!, container: String!, target_list: String!, only_if_ready: Boolean): String
    produce_artifacts(step: String!, paths: String, only_if_ready: Boolean): String

    {%- for rank in structure.keys() %}
    {{ rank.name }}{{ rank | rank_param }}: {{ rank.name | capitalize }}!
    {%- endfor %}
}

type Mutation {
    upload_b64(input: String!, container: String!): Boolean!
    upload_file(file: Upload, container: String!): Boolean!
    run_analysis(step: String!, analysis: String!, container: String!, targets: String!): String!
    run_all_analyses: String!
    analyses: AnalysisMutations!
}

type AnalysisMutations {
    {%- for step in steps %}
    {%- if step.analyses_count() > 0 %}
    {{ step.name | snake_case }}: {{ step.name }}Analyses!
    {%- endif %}
    {%- endfor %}
}

type Info {
    kinds: [Kind!]!
    ranks: [Rank!]!
    steps: [Step!]!
    globals: [String!]!
    model: String!
}

type Kind {
    name: ID
    rank: String!
    parent: String
}

type Rank {
    name: ID
    depth: Int!
    parent: String
}

type Step {
    name: ID
    parent: String
    containers: [Container!]!
    analyses: [Analysis!]!
}

type Analysis {
    name: ID
    arguments: [AnalysisArgument!]!
}

type AnalysisArgument {
    name: ID
    acceptable_kinds: [Kind!]!
}

type Container {
    name: ID
    mime: String
    targets: [Target!]!
}

type Target {
    serialized: String
    exact: Boolean
    path_components: [String!]!
    kind: String
    ready: Boolean
}

{% for rank, steps in structure.items() %}
type {{ rank.name | capitalize }} {
{%- for step in steps %}
    {{ step.name | snake_case }}(only_if_ready: Boolean): String!
{%- endfor %}
}
{% endfor %}

{% for step in steps %}
{%- if step.analyses_count() > 0 %}
type {{ step.name }}Analyses {
    {%- for analysis in step.analyses() %}
    {{ analysis.name | snake_case }}({{ analysis | generate_analysis_parameters }}): String!
    {%- endfor %}
}
{%- endif %}
{%- endfor %}


scalar Upload
