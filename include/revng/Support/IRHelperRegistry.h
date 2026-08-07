#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <string>

#include "revng/Support/ManagedStaticRegistry.h"

/// The name of a helper, as recorded in the registry
///
/// \note this is an implementation detail of `IRHelper`, which is how helpers
///       are declared and reached. The registry exists to reject two helpers
///       sharing a name.
struct IRHelperName {
public:
  std::string Name;

public:
  const std::string &key() const { return Name; }
};

using RegisterIRHelper = RegisterManagedStaticImpl<IRHelperName>;
