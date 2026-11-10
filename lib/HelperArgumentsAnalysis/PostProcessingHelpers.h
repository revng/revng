#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/IR/Module.h"

#include "revng/Model/FunctionTags.h"
#include "revng/Support/IRHelpers.h"

/// \return true for the variables introduced by rev.ng, such as the
///         `revng.const.*` strings, as opposed to QEMU's.
inline bool isSpecialGV(llvm::GlobalVariable &GV) {
  llvm::StringRef Name = GV.getName();
  return Name == "cpu_loop_exiting" or Name == "arch_cpu_type_beacon"
         or Name.starts_with(DefaultStringNamespace);
}

inline bool isHelper(llvm::Function &F) {
  return F.getName().starts_with("helper_");
}

inline void tagQEMUGlobalsAndFunctions(llvm::Module &Module) {
  // Tag all global objects in HelpersModule as QEMU
  for (llvm::GlobalVariable &G : Module.globals())
    FunctionTags::QEMU.addTo(&G);

  for (llvm::Function &F : Module.functions()) {
    if (F.isIntrinsic())
      continue;

    FunctionTags::QEMU.addTo(&F);

    if (F.hasFnAttribute(llvm::Attribute::NoReturn)
        or F.getSection() == "revng_exceptional")
      FunctionTags::Exceptional.addTo(&F);

    if (F.getName().starts_with("helper_"))
      FunctionTags::Helper.addTo(&F);
  }
}
