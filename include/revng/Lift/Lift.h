#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/StringRef.h"
#include "llvm/Pass.h"

#include "revng/PipeboxCommon/BinariesContainer.h"
#include "revng/PipeboxCommon/LLVMContainer.h"
#include "revng/PipeboxCommon/Model.h"
#include "revng/Support/Debug.h"

namespace KillReason {

enum Values {
  NonKiller,
  KillerSyscall,
  EndlessLoop,
  LeadsToKiller
};

inline llvm::StringRef getName(Values Reason) {
  switch (Reason) {
  case NonKiller:
    return "NonKiller";
  case KillerSyscall:
    return "KillerSyscall";
  case EndlessLoop:
    return "EndlessLoop";
  case LeadsToKiller:
    return "LeadsToKiller";
  }

  revng_abort("Unexpected reason");
}

inline Values fromName(llvm::StringRef Name) {
  if (Name == "NonKiller")
    return NonKiller;
  if (Name == "KillerSyscall")
    return KillerSyscall;
  else if (Name == "EndlessLoop")
    return EndlessLoop;
  else if (Name == "LeadsToKiller")
    return LeadsToKiller;
  else
    revng_abort("Unexpected name");
}

} // namespace KillReason

namespace revng::pypeline::piperuns {

class Lift {
public:
  static constexpr llvm::StringRef Name = "lift";
  using Arguments = TypeList<
    PipeRunArgument<const BinariesContainer, "Input", "Input binaries to lift">,
    PipeRunArgument<LLVMRootContainer,
                    "Output",
                    "LLVM Module containing the lifted binaries",
                    Access::Write>>;

private:
  const Model &TheModel;
  const BinariesContainer &Binary;
  LLVMRootContainer &ModuleContainer;

public:
  Lift(const class Model &Model,
       llvm::StringRef Config,
       llvm::StringRef DynamicConfig,
       const BinariesContainer &Binary,
       LLVMRootContainer &ModuleContainer);

  CustomInvalidationData run();

public:
  static llvm::Error checkPrecondition(const class Model &Model);

  static bool requiresCustomInvalidation(const ModelDiff &Diff);

  static std::vector<std::set<ObjectID>>
  processCustomInvalidation(const InvalidationData &Data,
                            const ModelDiff &Diff);
};

} // namespace revng::pypeline::piperuns
