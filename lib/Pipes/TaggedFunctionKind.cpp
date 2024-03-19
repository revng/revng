/// \file CompileModule.cpp
/// The isolated kind is used to rappresent isolated root and functions.

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <optional>
#include <string>

#include "llvm/ADT/STLExtras.h"

#include "revng/Model/Binary.h"
#include "revng/Model/IRHelpers.h"
#include "revng/Pipeline/AllRegistries.h"
#include "revng/Pipeline/Context.h"
#include "revng/Pipeline/Target.h"
#include "revng/Pipes/FileContainer.h"
#include "revng/Pipes/FunctionKind.h"
#include "revng/Pipes/Kinds.h"
#include "revng/Pipes/ModelGlobal.h"
#include "revng/Pipes/RootKind.h"
#include "revng/Pipes/TaggedFunctionKind.h"
#include "revng/Support/Assert.h"
#include "revng/Support/FunctionTags.h"
#include "revng/Support/MetaAddress.h"

using namespace pipeline;
using namespace ::revng::kinds;
using namespace llvm;

std::optional<pipeline::Target>
TaggedFunctionKind::symbolToTarget(const llvm::Function &Symbol) const {
  if (not Tag->isExactTagOf(&Symbol) or Symbol.isDeclaration())
    return std::nullopt;

  auto Address = getMetaAddressOfIsolatedFunction(Symbol);
  revng_assert(Address.isValid());
  return pipeline::Target({ Address.toString() }, *this);
}

void TaggedFunctionKind::appendAllTargets(const pipeline::Context &Ctx,
                                          pipeline::TargetsList &Out) const {
  const auto &Model = getModelFromContext(Ctx);
  for (const auto &Function : Model->Functions()) {
    Out.push_back(Target(Function.Entry().toString(), *this));
  }
}

cppcoro::generator<std::pair<const model::Function *, llvm::Function *>>
TaggedFunctionKind::getFunctionsAndCommit(pipeline::ExecutionContext &Context,
                                          llvm::Module &Module,
                                          llvm::StringRef ContainerName) {
  std::map<MetaAddress, llvm::Function *> AddressToFunction;

  for (llvm::Function &Function : Module.functions()) {
    if (not FunctionTags::Isolated.isTagOf(&Function)) {
      continue;
    }

    AddressToFunction.emplace(getMetaAddressOfIsolatedFunction(Function),
                              &Function);
  }
  auto &Binary = getModelFromContext(Context);

  for (const Target &Target :
       Context.getCurrentRequestedTargets()[ContainerName]) {
    auto MetaAddress = MetaAddress::fromString(Target.getPathComponents()[0]);
    Context.clearAndResumeTracking();
    auto &ModelFunction = Binary->Functions().at(MetaAddress);
    auto Iter = AddressToFunction.find(MetaAddress);
    if (Iter == AddressToFunction.end())
      co_yield std::pair<const model::Function *,
                         llvm::Function *>(&ModelFunction, nullptr);
    else
      co_yield std::pair<const model::Function *,
                         llvm::Function *>(&ModelFunction, Iter->second);
    Context.commit(Target, ContainerName);
  }
}

cppcoro::generator<const model::Function *>
TaggedFunctionKind::getFunctionsAndCommit(pipeline::ExecutionContext &Context,
                                          const pipeline::ContainerBase
                                            &Container) {
  auto &Binary = getModelFromContext(Context);
  for (const Target &Target :
       Context.getCurrentRequestedTargets()[Container.name()]) {
    Context.clearAndResumeTracking();

    auto MetaAddress = MetaAddress::fromString(Target.getPathComponents()[0]);
    co_yield &Binary->Functions().at(MetaAddress);

    Context.commit(Target, Container);
  }
}
