#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Pipeline/Runner.h"

namespace pipeline {

class InvalidationEventBase {
private:
  const char *ID;

public:
  explicit InvalidationEventBase(const char &ID) : ID(&ID) {}
  llvm::Error apply(Runner &Runner) const;
  void
  getInvalidations(const Runner &Runner, Runner::InvalidationMap &Out) const;
  virtual ~InvalidationEventBase() = default;

  const char *getID() const { return ID; }
};

template<typename Derived>
class InvalidationEvent : public InvalidationEventBase {
public:
  explicit InvalidationEvent() : InvalidationEventBase(Derived::ID) {}

  static bool classof(const InvalidationEventBase *Base) {
    return Base->getID() == &Derived::ID;
  }
};

} // namespace pipeline
