#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Model/DisassemblyConfiguration.h"
#include "revng/Model/NamingConfiguration.h"

#include "revng/Model/Generated/Early/Configuration.h"

namespace model {
class VerifyHelper;
}

class model::Configuration : public model::generated::Configuration {
public:
  using generated::Configuration::Configuration;

  bool verify(VerifyHelper &VH) const;
  bool verify(bool Assert) const debug_function;
  bool verify() const debug_function;
};

#include "revng/Model/Generated/Late/Configuration.h"
