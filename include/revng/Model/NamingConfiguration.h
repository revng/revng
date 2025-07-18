#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

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

  llvm::StringRef unnamedDynamicFunctionPrefix() const {
    if (UnnamedDynamicFunctionPrefix().empty())
      return "dynamic_function_";
    else
      return UnnamedDynamicFunctionPrefix();
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

  llvm::StringRef unnamedLocalVariablePrefix() const {
    if (UnnamedLocalVariablePrefix().empty())
      return "var_";
    else
      return UnnamedLocalVariablePrefix();
  }
  llvm::StringRef unnamedBreakFromLoopVariablePrefix() const {
    if (UnnamedBreakFromLoopVariablePrefix().empty())
      return "break_from_loop_";
    else
      return UnnamedBreakFromLoopVariablePrefix();
  }
  llvm::StringRef unnamedGotoLabelPrefix() const {
    if (UnnamedGotoLabelPrefix().empty())
      return "label_";
    else
      return UnnamedGotoLabelPrefix();
  }

  llvm::StringRef undefinedValuePrefix() const {
    if (UndefinedValuePrefix().empty())
      return "undef_";
    else
      return UndefinedValuePrefix();
  }
  llvm::StringRef opaqueCSVValuePrefix() const {
    if (OpaqueCSVValuePrefix().empty())
      return "revng_undefined_";
    else
      return OpaqueCSVValuePrefix();
  }
  llvm::StringRef maximumEnumValuePrefix() const {
    if (MaximumEnumValuePrefix().empty())
      return "enum_max_value_";
    else
      return MaximumEnumValuePrefix();
  }

  llvm::StringRef stackFrameVariableName() const {
    if (StackFrameVariableName().empty())
      return "stack";
    else
      return StackFrameVariableName();
  }
  llvm::StringRef rawStackArgumentName() const {
    if (RawStackArgumentName().empty())
      return "stack_arguments";
    else
      return RawStackArgumentName();
  }
  llvm::StringRef loopStateVariableName() const {
    if (LoopStateVariableName().empty())
      return "loop_state_var";
    else
      return LoopStateVariableName();
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
};

#include "revng/Model/Generated/Late/NamingConfiguration.h"
