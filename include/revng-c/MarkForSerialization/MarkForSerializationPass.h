#pragma once

//
// Copyright (c) rev.ng Srls. See LICENSE.md for details.
//

/// \brief Pass that identifies which Instructions must be serialized

#include <map>
#include <set>
#include <type_traits>

#include "llvm/Pass.h"

#include "revng/Support/Debug.h"

#include "MarkForSerializationFlags.h"

namespace llvm {

class Function;
class LoadInst;

} // end namespace llvm

extern Logger<> MarkLog;

struct MarkForSerializationPass : public llvm::FunctionPass {
public:
  static char ID;

  MarkForSerializationPass() : llvm::FunctionPass(ID) {}

  void getAnalysisUsage(llvm::AnalysisUsage &AU) const override;

  bool runOnFunction(llvm::Function &F) override;

  const auto &getMap() const { return ToSerialize; }

private:
  SerializationMap ToSerialize;
};
