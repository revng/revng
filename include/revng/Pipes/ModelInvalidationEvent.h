#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Model/Binary.h"
#include "revng/Pipeline/InvalidationEvent.h"
#include "revng/TupleTree/TupleTreeDiff.h"

namespace revng::pipes {

class ModelInvalidationEvent : public pipeline::InvalidationEvent {
public:
  explicit ModelInvalidationEvent(TupleTreeDiff<model::Binary> Diff) :
    Diff(std::move(Diff)) {}

  template<typename... Args>
  explicit ModelInvalidationEvent(Args &&...Arguments) :
    Diff(std::forward<Args>(Arguments)...) {}

  ModelInvalidationEvent() = delete;
  ModelInvalidationEvent(const ModelInvalidationEvent &Other) = delete;
  ModelInvalidationEvent(ModelInvalidationEvent &&Other) = delete;

  void getInvalidations(const pipeline::Runner &Runner,
                        pipeline::Runner::InvalidationMap &) const override;
  ~ModelInvalidationEvent() override = default;

private:
  TupleTreeDiff<model::Binary> Diff;
};
} // namespace revng::pipes
