{#
  This file is distributed under the MIT License. See LICENSE.md for details.
#}
type Query {
    info: Info!
    step(name: String!): Step
    container(name: String!, step: String!): Container
    target(step: String!, container: String!, target: String!): Target
    targets(pathspec: String!): [Step!]!
    produce(step: String!, container: String!, targetList: String!, onlyIfReady: Boolean): String
    produceArtifacts(step: String!, paths: String, onlyIfReady: Boolean): String

    {%- for rank in rank_to_artifact_steps.keys() %}
    {{ rank.name }}{{ rank | rank_param }}: {{ rank.name | capitalize }}!
    {%- endfor %}
}

type Mutation {
    uploadB64(input: String!, container: String!): Boolean!
    uploadFile(file: Upload, container: String!): Boolean!
    runAnalysis(step: String!, analysis: String!, container: String!, targets: String!): String!
    runAllAnalyses: String!
    analyses: AnalysisMutations!
    setGlobal(name: String!, content: String!, verify: Boolean): Boolean!
    applyDiff(globalName: String!, content: String!, verify: Boolean): Boolean!
}

type Subscription {
    invalidations: String!
}

type AnalysisMutations {
    {%- for step in steps %}
    {%- if step.analyses_count() > 0 %}
    {{ step.name }}: {{ step.name }}Analyses!
    {%- endif %}
    {%- endfor %}
}

type Info {
    kinds: [Kind!]!
    ranks: [Rank!]!
    steps: [Step!]!
    globals: [Global!]!
    global(name: String!): String!
    verifyGlobal(name: String!, content: String!): Boolean!
    verifyDiff(globalName: String!, content: String!): Boolean!
    model: String!
}

type Global {
    name: ID
    content: String!
}

type Kind {
    name: ID
    rank: String!
    parent: String
    definedLocations: [Rank!]!
    preferredKinds: [String!]!
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
    artifacts: Artifacts
}

type Artifacts {
    kind: Kind!
    container: Container!
    singleTargetFilename: String!
}

type Analysis {
    name: ID
    arguments: [AnalysisArgument!]!
}

type AnalysisArgument {
    name: ID
    acceptableKinds: [Kind!]!
}

type Container {
    name: ID
    mime: String
    targets: [Target!]!
}

type Target {
    serialized: String
    exact: Boolean
    pathComponents: [String!]!
    kind: String
    ready: Boolean
}

{% for rank, steps in rank_to_artifact_steps.items() %}
type {{ rank.name | capitalize }} {
{%- for step in steps %}
    {{ step.name }}(onlyIfReady: Boolean): String!
{%- endfor %}
}
{% endfor %}

{% for step in steps %}
{%- if step.analyses_count() > 0 %}
type {{ step.name }}Analyses {
    {%- for analysis in step.analyses() %}
    {{ analysis.name | normalize }}{{ analysis | generate_analysis_parameters }}: String!
    {%- endfor %}
}
{%- endif %}
{%- endfor %}


scalar Upload
