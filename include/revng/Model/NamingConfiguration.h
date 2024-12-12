#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

/* TUPLE-TREE-YAML

name: NamingConfiguration
type: struct
fields:
  - name: UnnamedSegmentPrefix
    doc: |
      The prefix for a segment without a name.

      The default value is `segment_`.
    type: string
    optional: true

  - name: UnnamedFunctionPrefix
    doc: |
      The prefix for a local function without a name.

      The default value is `function_`.
    type: string
    optional: true

  - name: UnnamedTypeDefinitionPrefix
    doc: |
      The prefix for a type definition without a name.

      The default value is "".

      Note that the type kind (like `struct`, or `typedef`) is going to be
      inserted automatically after this prefix.
    type: string
    optional: true

  - name: UnnamedEnumEntryPrefix
    doc: |
      The prefix for an enum entry without a name.

      The default value is `enum_entry_`.
    type: string
    optional: true
  - name: UnnamedStructFieldPrefix
    doc: |
      The prefix for a struct field without a name.

      The default value is `offset_`.
    type: string
    optional: true
  - name: UnnamedUnionFieldPrefix
    doc: |
      The prefix for a union member without a name.

      The default value is `member_`.
    type: string
    optional: true

  - name: UnnamedFunctionArgumentPrefix
    doc: |
      The prefix for a cabi function argument without a name.

      The default value is `argument_`.
    type: string
    optional: true
  - name: UnnamedFunctionRegisterPrefix
    doc: |
      The prefix for a raw function register without a name.

      The default value is `register_`.
    type: string
    optional: true

  - name: StructPaddingPrefix
    doc: |
      The prefix for a padding struct field.

      The default value is `padding_at_`.
    type: string
    optional: true
  - name: ArtificialReturnValuePrefix
    doc: |
      The prefix for an artificial raw function return value.

      The default value is `artificial_struct_returned_by_`.
    type: string
    optional: true
  - name: ArtificialArrayWrapperPrefix
    doc: |
      The prefix for an artificial array wrapper.

      The default value is `artificial_wrapper_`.
    type: string
    optional: true
  - name: ArtificialArrayWrapperFieldName
    doc: |
      The name of the field within the artificial array wrapper.
      See `ArtificialArrayWrapperPrefix`.

      The default value is `the_array`.
    type: string
    optional: true

  - name: ReserveNamesStartingWithUnderscore
    doc: |
      When this is set to `true`, all the names starting with underscores will
      have a \ref ReservedNamePrefix prefix attached.
    type: bool
    optional: true

TUPLE-TREE-YAML */

#include "revng/Model/Generated/Early/NamingConfiguration.h"

class model::NamingConfiguration
  : public model::generated::NamingConfiguration {
public:
  using generated::NamingConfiguration::NamingConfiguration;

  // TODO: remove these after the support the default TTG values is added.

  llvm::StringRef unnamedSegmentPrefix() const {
    if (UnnamedSegmentPrefix().empty())
      return "segment_";
    else
      return UnnamedSegmentPrefix();
  }

  llvm::StringRef unnamedFunctionPrefix() const {
    if (UnnamedFunctionPrefix().empty())
      return "function_";
    else
      return UnnamedFunctionPrefix();
  }

  llvm::StringRef unnamedTypeDefinitionPrefix() const {
    if (UnnamedTypeDefinitionPrefix().empty())
      return "";
    else
      return UnnamedTypeDefinitionPrefix();
  }

  llvm::StringRef unnamedEnumEntryPrefix() const {
    if (UnnamedEnumEntryPrefix().empty())
      return "enum_entry_";
    else
      return UnnamedEnumEntryPrefix();
  }
  llvm::StringRef unnamedStructFieldPrefix() const {
    if (UnnamedStructFieldPrefix().empty())
      return "offset_";
    else
      return UnnamedStructFieldPrefix();
  }
  llvm::StringRef unnamedUnionFieldPrefix() const {
    if (UnnamedUnionFieldPrefix().empty())
      return "member_";
    else
      return UnnamedUnionFieldPrefix();
  }

  llvm::StringRef unnamedFunctionArgumentPrefix() const {
    if (UnnamedFunctionArgumentPrefix().empty())
      return "argument_";
    else
      return UnnamedFunctionArgumentPrefix();
  }
  llvm::StringRef unnamedFunctionRegisterPrefix() const {
    if (UnnamedFunctionRegisterPrefix().empty())
      return "register_";
    else
      return UnnamedFunctionRegisterPrefix();
  }

  llvm::StringRef structPaddingPrefix() const {
    if (StructPaddingPrefix().empty())
      return "padding_at_";
    else
      return StructPaddingPrefix();
  }
  llvm::StringRef artificialReturnValuePrefix() const {
    if (ArtificialReturnValuePrefix().empty())
      return "artificial_struct_returned_by_";
    else
      return ArtificialReturnValuePrefix();
  }
  llvm::StringRef artificialArrayWrapperPrefix() const {
    if (ArtificialArrayWrapperPrefix().empty())
      return "artificial_wrapper_";
    else
      return ArtificialArrayWrapperPrefix();
  }
  llvm::StringRef artificialArrayWrapperFieldName() const {
    if (ArtificialArrayWrapperFieldName().empty())
      return "the_array";
    else
      return ArtificialArrayWrapperFieldName();
  }
};

#include "revng/Model/Generated/Late/NamingConfiguration.h"
