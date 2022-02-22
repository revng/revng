#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Pipeline/Runner.h"

namespace pipeline {

class InvalidationEvent {
public:
  llvm::Error run(Runner &Runner) const;
  virtual void getInvalidations(const Runner &Runner,
                                Runner::InvalidationMap &Out) const = 0;
  virtual ~InvalidationEvent() = default;
};

} // namespace pipeline
