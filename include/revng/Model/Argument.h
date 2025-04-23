#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/ADT/RecursiveCoroutine.h"
#include "revng/Model/Type.h"

#include "revng/Model/Generated/Early/Argument.h"

namespace model {
class VerifyHelper;
}

class model::Argument : public model::generated::Argument {
public:
  using generated::Argument::Argument;
  Argument(uint64_t Index, UpcastableType &&Type) :
    model::generated::Argument(Index, std::move(Type), {}, {}) {}

public:
  bool verify() const debug_function;
  bool verify(bool Assert) const debug_function;
  RecursiveCoroutine<bool> verify(VerifyHelper &VH) const;
};

#include "revng/Model/Generated/Late/Argument.h"
