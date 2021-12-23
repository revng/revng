#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Pipeline/LLVMContainer.h"
#include "revng/Pipeline/LLVMPipe.h"

namespace Pipeline {

/// A pure llvm pipe is a pipe that can be used to test out things quickly
/// when a pass has not tangible side effect on targets.
/// It has a empty contract and can be loaded with llvm passess directly.
class PureLLVMPipe {
private:
  std::vector<std::string> PassNames;

public:
  static constexpr auto Name = "PureLLVMPipe";

  std::vector<ContractGroup> getContract() const { return {}; }
  void run(const Context &, LLVMContainer &Container);

  template<typename OStream>
  void dump(OStream &OS, size_t Indents = 0) const debug_function {
    for (const auto &Pass : PassNames) {
      indent(OS, Indents);
      OS << Pass << "\n";
    }
  }

  static llvm::Expected<PureLLVMPipe>
  create(std::vector<std::string> PassNames) {
    for (const auto &Name : PassNames)
      if (llvm::PassRegistry::getPassRegistry()->getPassInfo(Name) == nullptr)
        return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                       "Could not load llvm pass %s ",
                                       Name.c_str());

    return PureLLVMPipe(std::move(PassNames));
  }

private:
  PureLLVMPipe(std::vector<std::string> Names) : PassNames(std::move(Names)) {}
};

template<typename... LLVMPasses>
PipeWrapper wrapLLVMPasses(std::string LLVMModuleName, LLVMPasses &&...P) {
  return PipeWrapper(LLVMPipe(std::move(P)...), { std::move(LLVMModuleName) });
}
} // namespace Pipeline
