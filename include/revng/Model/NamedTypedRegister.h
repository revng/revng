#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/ADT/RecursiveCoroutine.h"
#include "revng/Model/Type.h"

#include "revng/Model/Generated/Early/NamedTypedRegister.h"

namespace model {
class VerifyHelper;
}

class model::NamedTypedRegister : public model::generated::NamedTypedRegister {
public:
  using generated::NamedTypedRegister::NamedTypedRegister;

public:
  bool verify() const debug_function;
  bool verify(bool Assert) const debug_function;
  RecursiveCoroutine<bool> verify(VerifyHelper &VH) const;
};

#include "revng/Model/Generated/Late/NamedTypedRegister.h"
