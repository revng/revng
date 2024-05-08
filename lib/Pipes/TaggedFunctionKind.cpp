/// \file CompileModule.cpp
/// The isolated kind is used to rappresent isolated root and functions.

//
// This file is distributed under the MIT License. See LICENSE.mit for details.
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
