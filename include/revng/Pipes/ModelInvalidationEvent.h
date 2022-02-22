#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Model/Binary.h"
#include "revng/Pipeline/InvalidationEvent.h"
#include "revng/TupleTree/TupleTreeDiff.h"

namespace revng::pipes {

class ModelInvalidationEvent
  : public pipeline::InvalidationEvent<ModelInvalidationEvent> {
private:
  TupleTreeDiff<model::Binary> Diff;

public:
  static char ID;

public:
  explicit ModelInvalidationEvent(TupleTreeDiff<model::Binary> &&Diff) :
    Diff(std::move(Diff)) {}

  explicit ModelInvalidationEvent(const TupleTreeDiff<model::Binary> &Diff) :
    Diff(Diff) {}

public:
  ~ModelInvalidationEvent() override = default;

  const TupleTreeDiff<model::Binary> getDiff() const { return Diff; }
};
} // namespace revng::pipes
