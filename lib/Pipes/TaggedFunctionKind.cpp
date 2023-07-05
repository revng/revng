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
  if (not Tag->isTagOf(&Symbol) or Symbol.isDeclaration())
    return std::nullopt;

  auto Address = getMetaAddressOfIsolatedFunction(Symbol);
  revng_assert(Address.isValid());
  return pipeline::Target({ Address.toString() }, *this);
}

using TaggedFK = TaggedFunctionKind;
void TaggedFK::getInvalidations(const Context &Ctx,
                                TargetsList &ToRemove,
                                const GlobalTupleTreeDiff &Diff) const {
  const auto &CurrentModel = getModelFromContext(Ctx);

  if (not Ctx.containsReadOnlyContainer(BinaryCrossRelationsRole))
    return;

  const auto *ModelDiff = Diff.getAs<model::Binary>();
  if (not ModelDiff)
    return;
}

void TaggedFunctionKind::appendAllTargets(const pipeline::Context &Ctx,
                                          pipeline::TargetsList &Out) const {
  const auto &Model = getModelFromContext(Ctx);
  for (const auto &Function : Model->Functions()) {
    Out.push_back(Target(Function.Entry().toString(), *this));
  }
}
