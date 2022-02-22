#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/ArrayRef.h"

#include "revng/Pipeline/Container.h"
#include "revng/Pipeline/Runner.h"
#include "revng/Pipeline/Target.h"

namespace revng::pipes {

class ModelInvalidationRule {
public:
  ModelInvalidationRule() { getRulesRegistry().push_back(this); }

  static llvm::ArrayRef<ModelInvalidationRule *> getRules() {
    return getRulesRegistry();
  }

  virtual void registerInvalidations(const pipeline::ContainerBase &Container,
                                     pipeline::TargetsList &ToRemove) const = 0;

  static void
  registerAllInvalidations(const pipeline::Runner &Runner,
                           pipeline::Runner::InvalidationMap &Invalidations);

  virtual ~ModelInvalidationRule() = default;

private:
  static std::vector<ModelInvalidationRule *> &getRulesRegistry() {
    static std::vector<ModelInvalidationRule *> Rules;
    return Rules;
  }
};

} // namespace revng::pipes
