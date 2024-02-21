/**
 * This file is distributed under the MIT License. See LICENSE.md for details.
 */
/* eslint-disable @typescript-eslint/naming-convention */

import * as yaml from "yaml";
import { deepEqual } from "fast-equals";
import { yamlParseOptions, yamlOutParseOptions, yamlToStringOptions } from "./tuple_tree";
import { _getElementByPath, _setElementByPath, _getTypeInfo, _makeDiff, _validateDiff, _applyDiff, BigIntBuilder, DiffSet, TypeInfo, TypeHints, IReference, Reference } from "./tuple_tree";
export { DiffSet, IReference, Reference };

{% for file_name in external_files %}
{{ file_name | read_file }}
{% endfor %}

export function parseI{{ global_name }}(input: string): I{{ global_name }} {
  return yaml.parse(input, yamlParseOptions);
}

export function parse{{ global_name }}(input: string): {{ global_name }} {
  return new {{ global_name }}(parseI{{ global_name }}(input));
}

export function dump{{ global_name }}(tuple_tree: I{{ global_name }} | {{ global_name }}): string {
  const doc = new yaml.Document(tuple_tree, yamlOutParseOptions);
  return doc.toString(yamlToStringOptions);
}

export function clone(tuple_tree: {{ global_name }}): {{ global_name }} {
    return parse{{ global_name }}(dump{{ global_name }}(tuple_tree));
}

{% for type_name in string_types %}
export type I{{ type_name }} = string;

export class {{ type_name }} {
    str: string;

    constructor(str: string) {
        this.str = str;
    }

    toJSON(): string {
        return this.str;
    }

    toString(): string {
        return this.str;
    }

    equals(other: unknown): boolean {
        if (other instanceof {{ type_name}}) {
            return this.str == other.str;
        } else {
            return false;
        }
    }

    static parse(str?: string): {{ type_name }} {
        return new {{ type_name }}(str || "");
    }
}
{% endfor %}

{% for enum in enums %}
{{ enum.doc | ts_doc }}
export const {{enum.name}}Values = [
    "Invalid",
    {%- for member in enum.members %}
        {%- if member.doc %}
        {{ member.doc | ts_doc }}
        {%- endif %}
        "{{ member.name }}",
    {%- endfor %}
] as const;
export type {{enum.name}} = typeof {{enum.name}}Values[number];
{% endfor %}


{% for class_ in structs %}
{{ class_.doc | ts_doc }}
export interface I{{class_.name}} {% if class_.inherits %} extends I{{class_.inherits.name}} {% endif %} {
    {%- for field in class_.fields %}
    {%- if field.doc %}
    {{ field.doc | ts_doc }}
    {%- endif %}
    {{field.name}}{{'?' if is_optional(field) else ''}}: {{field | ts_itype }}
    {%- endfor %}
}

{% if (not class_.abstract) and class_.inherits %}
export function isI{{class_.name}}(obj: I{{class_.inherits.name}}): obj is I{{class_.name}} {
    return obj.Kind == "{{class_.name}}";
}
{%- endif %}

export {% if class_.abstract %}abstract{% endif %} class {{class_.name}} {% if class_.inherits %} extends {{class_.inherits.name}} {% endif %} {
    {%- for field in class_.fields %}
    {%- if field.doc %}
    {{ field.doc | ts_doc }}
    {%- endif %}
    {{field.name}}: {{ field | ts_type }}
    {%- endfor %}

    constructor(
        rawObject{{ '?' if completely_optional(class_) else '' }}: I{{class_.name}}
        {%- if class_ | get_guid %}
        ,genGuid: (rawObject: I{{class_.name}}) => {{ class_ | get_guid | ts_type }}) {
            if (rawObject.{{class_ | get_guid | get_attr('name') }} === undefined) {
                rawObject.{{class_ | get_guid | get_attr('name') }} = genGuid(rawObject);
            }
        {%- else %}
        ) {
        {%- endif %}
        {%- if completely_optional(class_) %}
        if (rawObject === undefined) {
            rawObject = {};
        }
        {%- endif %}
        {%- if class_.inherits %}
        {%- if class_.inherits | get_guid %}
        super(rawObject, gen{{class_.name}}Guid);
        {%- else %}
        super(rawObject)
        {%- endif %}
        {%- endif %}
        {%- for field in class_.fields %}
            {{ field | gen_assignment }}
        {%- endfor %}
    }

    {% if class_.abstract %}
    static parse(rawObject: I{{class_.name}}): {{class_.name}} {
        switch(rawObject.Kind) {
        {%- for child in class_.children %}
        case "{{child.name}}":
            return new {{child.name}}(rawObject as I{{child.name}});
        {%- endfor %}
        case "Invalid":
            throw new Error("Invalid Kind")
        }
    }

    static parseClass(obj: {{class_.name}}) {
        switch(obj.Kind) {
        {%- for child in class_.children %}
        case "{{child.name}}":
            return {{child.name}};
        {%- endfor %}
        case "Invalid":
            throw new Error("Invalid Kind")
        }
    }

    static parseKey(key: string): {[key: string]: string} {
        const parts = key.split('-')
        return { {{ class_ | key_parser }}  };
    }
    {%- endif %}

    {% if class_.key_fields | length > 0 %}
    static keyed = true;
    key(): string {
        return {{ class_ | gen_key }};
    }
    {% endif %}

    public toJSON(): I{{ class_.name }} {
        const result = {{ "super.toJSON()" if class_.inherits else "{}" }};
        {%- for field in class_.fields %}
        {%- if is_optional(field) %}
        if (!deepEqual(this.{{ field.name }}, {{ default_value(field) }})) {
            result["{{ field.name }}"] = this.{{ field.name }};
        }
        {%- else %}
        result["{{ field.name }}"] = this.{{ field.name }};
        {%- endif %}
        {%- endfor %}
        return result as I{{ class_.name }};
    }

    static fromString(input: string): {{ class_.name }} {
        const object = yaml.parse(input, yamlParseOptions);
        {%- if class_.abstract %}
        return {{ class_.name }}.parse(object);
        {%- else %}
        return new {{ class_.name }}(object);
        {%- endif %}
    }

    dump(): string {
        const doc = new yaml.Document(this, yamlOutParseOptions);
        return doc.toString(yamlToStringOptions);
    }
}
{% endfor %}

export const TYPE_HINTS: TypeHints = new Map();
{% for class_ in structs %}
TYPE_HINTS.set({{class_.name}}, {
{%- for field in class_.fields %}
    {{field.name}}: {{ field | type_hint }}{% if not loop.last %},{% endif %}
{%- endfor %}
});
{% endfor %}

export function getTypeInfo(
  path: string | string[],
  root: any = {{ metadata.root }}
): TypeInfo | undefined {
  return _getTypeInfo(path, root, TYPE_HINTS);
}

export function makeDiff(tuple_tree_old: {{ global_name }}, tuple_tree_new: {{ global_name }}): DiffSet {
  return _makeDiff(tuple_tree_old, tuple_tree_new, TYPE_HINTS, {{ metadata.root }});
}

export function validateDiff(obj: {{ global_name }}, diffs: DiffSet): boolean {
  return _validateDiff(obj, diffs, getTypeInfo);
}

export function applyDiff(obj: {{ global_name }}, diffs: DiffSet): [false] | [true, {{ global_name }}] {
  return _applyDiff(obj, diffs, validateDiff, getTypeInfo, clone);
}

export function getElementByPath<T>(path: string, tree: {{ global_name }}): T | undefined {
  return _getElementByPath(path, tree);
}

export function setElementByPath<T>(path: string, tree: {{ global_name }}, value: T): boolean {
  return _setElementByPath(path, tree, value);
}

{% if global_name != metadata.root %}
export type I{{ global_name }} = I{{ metadata.root }};
export class {{ global_name }} extends {{ metadata.root }} {};
{% endif %}
