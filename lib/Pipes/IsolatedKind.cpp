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
#include "revng/Pipeline/Context.h"
#include "revng/Pipeline/PathComponent.h"
#include "revng/Pipeline/Registry.h"
#include "revng/Pipeline/Target.h"
#include "revng/Pipes/IsolatedKind.h"
#include "revng/Pipes/Kinds.h"
#include "revng/Pipes/RootKind.h"
#include "revng/Support/Assert.h"
#include "revng/Support/FunctionTags.h"
#include "revng/Support/MetaAddress.h"

using namespace pipeline;
using namespace revng;
using namespace revng::pipes;
using namespace llvm;

Rank pipes::FunctionsRank("Function Rank", RootRank);

TaggedFunctionKind pipes::Isolated("Isolated", FunctionTags::Lifted);
static RegisterKind K(Isolated);

void TaggedFunctionKind::expandTarget(const Context &Ctx,
                                      const Target &Input,
                                      TargetsList &Output) const {
  if (not Input.getPathComponents().back().isAll()) {
    Output.push_back(Input);
    return;
  }

  const auto &Model = getModelFromContext(Ctx);
  if (Model.Functions.empty()) {
    Output.push_back(Input);
    return;
  }

  for (const auto &Function : Model.Functions) {

    Target ToInsert({ Input.getPathComponents().front(),
                      PathComponent(Function.Entry.toString()) },
                    *this);
    Output.emplace_back(std::move(ToInsert));
  }
}
