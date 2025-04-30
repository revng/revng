#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <limits>
#include <string>

#include "revng/ADT/SortedVector.h"
#include "revng/Support/MetaAddress.h"
#include "revng/Support/MetaAddress/YAMLTraits.h"
#include "revng/Yield/BasicBlock.h"

#include "revng/Yield/Generated/Early/Function.h"

namespace model {
class VerifyHelper;
}

namespace yield {

class Function : public generated::Function {
public:
  using generated::Function::Function;

public:
  bool verify() const debug_function;
  bool verify(bool Assert) const debug_function;
  bool verify(model::VerifyHelper &VH) const;
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
