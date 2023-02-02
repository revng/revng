#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Pipes/Kinds.h"
#include "revng/Pipes/ModelGlobal.h"
#include "revng/Pipes/TaggedFunctionKind.h"

namespace revng::kinds {

class IsolatedFunctionKind : public TaggedFunctionKind {
public:
  using TaggedFunctionKind::TaggedFunctionKind;

  void getInvalidations(const pipeline::Context &Ctx,
                        pipeline::TargetsList &ToRemove,
                        const pipeline::GlobalTupleTreeDiff &Diff,
                        const pipeline::Global &Before,
                        const pipeline::Global &After) const override;
};

inline IsolatedFunctionKind
  Isolated("Isolated", ranks::Function, FunctionTags::Isolated);

} // namespace revng::kinds
