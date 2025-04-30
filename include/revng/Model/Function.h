#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/ADT/MutableSet.h"
#include "revng/ADT/SortedVector.h"
#include "revng/Model/CallSitePrototype.h"
#include "revng/Model/CommonFunctionMethods.h"
#include "revng/Model/FunctionAttribute.h"
#include "revng/Model/StatementComment.h"
#include "revng/Model/TypeDefinition.h"
#include "revng/Support/MetaAddress.h"
#include "revng/Support/MetaAddress/YAMLTraits.h"
#include "revng/TupleTree/TupleTree.h"

#include "revng/Model/Generated/Early/Function.h"

namespace model {
class VerifyHelper;
}

class model::Function : public model::generated::Function,
                        public model::CommonFunctionMethods<Function> {
public:
  using generated::Function::Function;

public:
  /// The helper for stack frame type unwrapping.
  /// Use this when you need to access/modify the existing struct,
  /// and \ref StackFrameType() when you need to assign a new one.
  model::StructDefinition *stackFrameType() {
    if (StackFrameType().isEmpty())
      return nullptr;
    else
      return StackFrameType()->getStruct();
  }

  /// The helper for stack argument type unwrapping.
  /// Use this when you need to access/modify the existing struct,
  /// and \ref StackFrameType() when you need to assign a new one.
  const model::StructDefinition *stackFrameType() const {
    if (StackFrameType().isEmpty())
      return nullptr;
    else
      return StackFrameType()->getStruct();
  }

public:
  bool verify() const debug_function;
  bool verify(bool Assert) const debug_function;
  bool verify(VerifyHelper &VH) const;
  void dumpTypeGraph(const char *Path,
                     const model::Binary &Binary) const debug_function;
};

#include "revng/Model/Generated/Late/Function.h"
