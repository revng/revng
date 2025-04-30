#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Model/DisassemblyConfigurationAddressStyle.h"
#include "revng/Model/DisassemblyConfigurationImmediateStyle.h"

#include "revng/Model/Generated/Early/DisassemblyConfiguration.h"

class model::DisassemblyConfiguration
  : public model::generated::DisassemblyConfiguration {
public:
  using generated::DisassemblyConfiguration::DisassemblyConfiguration;
};

#include "revng/Model/Generated/Late/DisassemblyConfiguration.h"
