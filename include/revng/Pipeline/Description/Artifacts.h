#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Pipeline/Description/Generated/Early/Artifacts.h"

class pipeline::description::Artifacts
  : public pipeline::description::generated::Artifacts {
public:
  using generated::Artifacts::Artifacts;

  bool isValid() {
    return not(this->Container().empty() or this->Kind().empty()
               or this->SingleTargetFilename().empty());
  }
};

#include "revng/Pipeline/Description/Generated/Late/Artifacts.h"
