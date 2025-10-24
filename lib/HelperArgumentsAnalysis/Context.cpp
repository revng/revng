/// \file Context.cpp

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/ADT/RecursiveCoroutine.h"

#include "Context.h"

namespace aua {

RecursiveCoroutine<const Value *>
Context::replaceArguments(const Value &Original,
                          const llvm::DenseMap<uint64_t, const Value *>
                            &NewArguments) {

  if (auto *Argument = llvm::dyn_cast<ArgumentValue>(&Original)) {
    auto It = NewArguments.find(Argument->index());
    if (It != NewArguments.end())
      rc_return It->second;
    else
      rc_return &Original;
  }

  SmallVector<const Value *, 2> NewOperandList;
  for (const Value *V : Original.Operands) {
    const Value *Replacement = rc_recur replaceArguments(*V, NewArguments);
    NewOperandList.push_back(Replacement);
  }

  if (NewOperandList == Original.Operands) {
    rc_return &Original;
  } else {
    rc_return Original.upcast([this, &NewOperandList](auto &Upcasted)
                                -> const Value * {
      auto Copy = Upcasted;
      Copy.Operands = NewOperandList;
      return &get(std::move(Copy));
    });
  }
}

} // namespace aua
