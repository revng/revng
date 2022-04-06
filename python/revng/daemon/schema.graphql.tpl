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
}

type Info {
    kinds: [Kind!]!
    ranks: [Rank!]!
    steps: [Step!]!
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

scalar Upload
