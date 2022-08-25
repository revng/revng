/// \file AllFunctions.cpp
/// \brief

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Model/Binary.h"
#include "revng/Model/Function.h"
#include "revng/Pipes/AllFunctions.h"
#include "revng/TupleTree/TupleTree.h"

namespace revng::kinds {

pipeline::TargetsList
compactFunctionTargets(const TupleTree<model::Binary> &Model,
                       pipeline::TargetsList::List &Targets,
                       const pipeline::Kind &K) {

  if (Model->Functions.size() == 0)
    return Targets;

  std::set<std::string> TargetsSet;
  const pipeline::Target AllFunctions({ pipeline::PathComponent::all() }, K);

  // enumerate the targets
  for (const pipeline::Target &Target : Targets) {
    // if we see a * then no need to check if we have all functions, just
    // return that.
    if (Target.getPathComponents().back().isAll())
      return pipeline::TargetsList({ AllFunctions });

    TargetsSet.insert(Target.getPathComponents().back().getName());
  }

  // Check if all functions in the model, that are not fake, are in the targets.
  // if they are, return *
  const auto IsInTargetSet = [&TargetsSet](const model::Function &F) {
    return TargetsSet.contains(F.Entry.toString());
  };
  if (llvm::all_of(Model->Functions, IsInTargetSet))
    return pipeline::TargetsList({ AllFunctions });

  return Targets;
}

} // namespace revng::kinds
