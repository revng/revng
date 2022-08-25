#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Model/Binary.h"
#include "revng/Pipeline/Kind.h"
#include "revng/Pipeline/Target.h"

namespace revng::kinds {

pipeline::TargetsList
compactFunctionTargets(const TupleTree<model::Binary> &Model,
                       pipeline::TargetsList::List &Targets,
                       const pipeline::Kind &K);

} // namespace revng::kinds
