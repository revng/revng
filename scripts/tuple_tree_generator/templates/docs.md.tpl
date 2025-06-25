<!--- - This template file is distributed under the MIT License. See LICENSE.md for details. - --->
<!--- - The notice below applies to the generated files. - --->

<!-- - macro references(type) - -->
<!-- - if references_map[type.name] -->
**Referenced by**: <!-- for struct, field in references_map[type.name] - -->
<!-- if loop.index > 1 -->, <!-- endif -->[`{{ struct.name }}.{{ field.name }}`](#{{ struct.name }}.{{ field.name }})
<!-- - endfor - -->
<!-- - endif -->
<!-- - endmacro - -->

<!-- - macro emit_index(struct, indentation, name) -->
{{ "  " * indentation * 2 }}- <!-- if name -->`{{name}}`: <!-- endif -->{{ struct | type_name }}

<!-- for field in struct.fields - -->
<!-- if field.resolved_type is struct -->
{{emit_index(field.resolved_type, indentation + 1, field.name)}}
<!-- elif field.resolved_type is upcastable -->
{{emit_index(field.resolved_type.base, indentation + 1, field.name)}}
<!-- elif field.resolved_type is sequence and field.resolved_type.element_type is struct -->
{{emit_index(field.resolved_type.element_type, indentation + 1, field.name)}}
<!-- elif field.resolved_type is sequence and field.resolved_type.element_type is upcastable -->
{{emit_index(field.resolved_type.element_type.base, indentation + 1, field.name)}}
<!-- endif -->
<!-- - endfor -->

<!-- endmacro - -->


<!-- - macro process_struct(struct, header) - -->
{{header}} <a id="{{ struct.name }}"></a>`{{ struct.name }}`

<!-- if struct.doc - -->{{struct.doc}}<!-- - endif -->

<!-- if struct.inherits -->
**Inherits from**: {{struct.inherits | type_name}}
<!-- endif -->

<!-- if struct.key_fields -->
**Key**: <!-- for field in struct.key_fields - --><!-- if loop.index > 1 -->, <!-- endif -->`{{field.name}}`<!-- endfor -->
<!-- endif -->

<!-- if struct.abstract -->
**Inheritors**: <!-- for inheritor in struct.inheritors - --><!-- if loop.index > 1 -->, <!-- endif -->{{inheritor | type_name}}<!-- endfor -->
<!-- endif -->

{{ references(struct) }}

<!-- if struct.abstract -->
<!-- for inheritor in struct.inheritors - -->
{{process_struct(inheritor, "####")}}
<!-- endfor -->

<!-- endif -->

**Fields**:

<!-- for field in struct.fields - -->
- <a id="{{ struct.name }}.{{ field.name }}"></a>`{{field.name}}` ({{field.resolved_type | type_name}}).

    <!-- if field.doc -->{{field.doc | indent(2) }}<!-- endif -->

<!-- endfor -->

<!-- endmacro - -->

<!-- - macro process_enum(enum) - -->
### <a id="{{ enum.name }}"></a>`{{ enum.name }}`

<!-- if enum.doc - -->{{enum.doc}}<!-- - endif -->

**Members**:

<!-- for member in enum.members -->
- `{{member.name}}`

    <!-- if member.doc -->{{member.doc | indent(2) }}<!-- endif -->
<!-- endfor -->

{{ references(enum) }}

<!-- endmacro - -->

## Data structures


<!-- for struct in structs - -->
<!-- - if struct.name == root_type - -->
Types reachable from the root:

{{ emit_index(struct, 0, "") }}
<!-- - endif - -->
<!-- endfor - -->

<!-- for struct in structs - -->
<!-- - if struct.abstract - -->
Inheritors of `{{struct.name}}`:
<!-- for inheritor in struct.inheritors -->
{{ emit_index(inheritor, 0, "") }}
<!-- endfor -->
<!-- - endif - -->
<!-- endfor - -->

<!-- for struct in structs - -->
<!-- - if struct.name == root_type - -->
{{ process_struct(struct, "###") }}
<!-- - endif - -->
<!-- endfor - -->

<!-- - for struct in structs - -->
<!-- - if struct.name != root_type - -->
<!-- if not struct.inherits -->
{{ process_struct(struct, "###") }}
<!-- - endif - -->
<!-- - endif - -->
<!-- endfor - -->

## Enumerations

<!-- for enum in enums - -->
{{ process_enum(enum) }}
<!-- - endfor - -->
