/// \file LLVMKind.cpp
/// \brief A llvm kind is a kind uses to extend the functionality of llvm
/// containers by providing new rules to associate targets to global objects

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/DenseSet.h"
#include "llvm/IR/GlobalValue.h"

#include "revng/Pipeline/LLVMContainer.h"
#include "revng/Pipeline/LLVMGlobalKindBase.h"
#include "revng/Pipeline/PureLLVMPipe.h"
#include "revng/Pipeline/Target.h"

using namespace pipeline;
using namespace llvm;

void PureLLVMPipe::run(const Context &, LLVMContainer &Container) {
  legacy::PassManager Manager;
  auto *Registry = PassRegistry::getPassRegistry();
  for (const auto &Element : PassNames) {

    Manager.add(Registry->getPassInfo(Element)->createPass());
  }
  Manager.run(Container.getModule());
}
