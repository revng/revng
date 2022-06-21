#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Model/Binary.h"
#include "revng/Pipeline/Kind.h"
#include "revng/Pipeline/PathComponent.h"
#include "revng/Pipeline/RegisterKind.h"
#include "revng/Pipeline/Target.h"
#include "revng/Pipes/ModelGlobal.h"

namespace revng::pipes {

inline void expandTargetImpl(const pipeline::Context &Ctx,
                             const pipeline::Target &Input,
                             pipeline::TargetsList &Output,
                             const pipeline::Kind &Kind) {
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
    auto Component = pipeline::PathComponent(Function.Entry.toString());
    pipeline::Target ToInsert({ Component }, Kind);
    Output.emplace_back(std::move(ToInsert));
  }
}

class FunctionKind : public pipeline::Kind {
public:
  using pipeline::Kind::Kind;

  void expandTarget(const pipeline::Context &Ctx,
                    const pipeline::Target &Input,
                    pipeline::TargetsList &Output) const override {
    expandTargetImpl(Ctx, Input, Output, *this);
  }
};
} // namespace revng::pipes
