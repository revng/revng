/// \file ModelInvalidationRule.cpp
/// \brief Implementation of model invalidation rules

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Pipes/ModelInvalidationRule.h"

using namespace revng::pipes;
using namespace pipeline;

using InvalidationMap = Runner::InvalidationMap;

void ModelInvalidationRule::registerAllInvalidations(const Runner &Runner,
                                                     InvalidationMap &Map) {
  for (const auto &Step : Runner) {
    auto &StepInvalidations = Map[Step.getName()];
    for (const auto &Cotainer : Step.containers()) {
      if (not Cotainer.second)
        continue;

      auto &ContainerInvalidations = StepInvalidations[Cotainer.first()];
      for (const auto *Rule : getRulesRegistry())
        Rule->registerInvalidations(*Cotainer.second, ContainerInvalidations);
    }
  }
}
