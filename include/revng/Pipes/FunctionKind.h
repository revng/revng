#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Model/Binary.h"
#include "revng/Pipeline/Kind.h"
#include "revng/Pipeline/RegisterKind.h"
#include "revng/Pipeline/Target.h"
#include "revng/Pipes/ModelGlobal.h"

namespace revng::kinds {

class FunctionKind : public pipeline::Kind {
public:
  using pipeline::Kind::Kind;
  void appendAllTargets(const pipeline::Context &Ctx,
                        pipeline::TargetsList &Out) const override {
    using namespace pipeline;
    const auto &Model = getModelFromContext(Ctx);
    for (const auto &Function : Model->Functions) {
      Out.push_back(Target(Function.Entry.toString(), *this));
    }
  }
};
} // namespace revng::kinds
