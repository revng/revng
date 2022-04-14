/// \file CompileModule.cpp
/// \brief The isolated kind is used to rappresent isolated root and functions.

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
#include "revng/Pipeline/PathComponent.h"
#include "revng/Pipeline/Target.h"
#include "revng/Pipes/Kinds.h"
#include "revng/Pipes/ModelGlobal.h"
#include "revng/Pipes/RootKind.h"
#include "revng/Pipes/TaggedFunctionKind.h"
#include "revng/Support/Assert.h"
#include "revng/Support/FunctionTags.h"
#include "revng/Support/MetaAddress.h"

using namespace pipeline;
using namespace ::revng::pipes;
using namespace llvm;

pipeline::TargetsList
TaggedFunctionKind::compactTargets(const pipeline::Context &Ctx,
                                   pipeline::TargetsList::List &Targets) const {

  const model::Binary &Model = *getModelFromContext(Ctx);

  if (Model.Functions.size() == 0) {
    return Targets;
  }

  std::set<std::string> TargetsSet;
  const pipeline::Target AllFunctions({ pipeline::PathComponent::all() },
                                      *this);

  // enumerate the targets
  for (const pipeline::Target &Target : Targets) {
    // if we see a * then no need to check if we have all functions, just
    // return that.
    if (Target.getPathComponents().back().isAll())
      return pipeline::TargetsList({ AllFunctions });

    TargetsSet.insert(Target.getPathComponents().back().getName());
  }

  // check if all functions in the model, that are not fake, are in the targets.
  // if they are, return *
  const auto IsNotFake = [](const auto &F) {
    return F.Type != model::FunctionType::Fake;
  };
  const auto IsInTargetSet = [&TargetsSet](const model::Function &F) {
    return TargetsSet.contains(F.Entry.toString());
  };
  if (llvm::all_of(llvm::make_filter_range(Model.Functions, IsNotFake),
                   IsInTargetSet)) {
    return pipeline::TargetsList({ AllFunctions });
  }

  return Targets;
}

void TaggedFunctionKind::expandTarget(const Context &Ctx,
                                      const Target &Input,
                                      TargetsList &Output) const {
  if (not Input.getPathComponents().back().isAll()) {
    Output.push_back(Input);
    return;
  }

  const model::Binary &Model = *getModelFromContext(Ctx);
  if (Model.Functions.empty()) {
    Output.push_back(Input);
    return;
  }

  for (const auto &Function : Model.Functions) {

    Target ToInsert({ PathComponent(Function.Entry.toString()) }, *this);
    Output.emplace_back(std::move(ToInsert));
  }
}

std::optional<pipeline::Target>
TaggedFunctionKind::symbolToTarget(const llvm::Function &Symbol) const {
  if (not Tag->isTagOf(&Symbol) or Symbol.isDeclaration())
    return std::nullopt;

  auto Address = getMetaAddressOfIsolatedFunction(Symbol);
  revng_assert(Address.isValid());
  return pipeline::Target({ Address.toString() }, *this);
}
