#pragma once

//
// This file is distributed under the MIT License. See LICENSE.mit for details.
//

#include <limits>
#include <string>

#include "revng/ADT/SortedVector.h"
#include "revng/Model/VerifyHelper.h"
#include "revng/Support/MetaAddress.h"
#include "revng/Support/MetaAddress/YAMLTraits.h"
#include "revng/Yield/BasicBlock.h"

/* TUPLE-TREE-YAML

name: Function
type: struct
fields:
  - name: Entry
    type: MetaAddress

  - name: ControlFlowGraph
    sequence:
      type: SortedVector
      elementType: BasicBlock

key:
  - Entry

TUPLE-TREE-YAML */

#include "revng/Yield/Generated/Early/Function.h"

namespace yield {

class Function : public generated::Function {
public:
  using generated::Function::Function;

public:
  bool verify(model::VerifyHelper &VH) const;
  void dump() const debug_function;

public:
  inline bool verify() const debug_function { return verify(false); }
  inline bool verify(bool Assert) const debug_function {
    model::VerifyHelper VH(Assert);
    return verify(VH);
  }
};

// TODO: move me somewhere more appropriate
inline const model::Function *tryGetFunction(const model::Binary &Binary,
                                             const BasicBlockID &Target) {
  const model::Function *F = nullptr;
  if (not Target.isInlined()) {
    auto It = Binary.Functions().find(Target.start());
    if (It != Binary.Functions().end())
      F = &*It;
  }

  return F;
}

} // namespace yield

#include "revng/Yield/Generated/Late/Function.h"
