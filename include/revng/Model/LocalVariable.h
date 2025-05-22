#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Support/MetaAddress.h"

#include "revng/Model/Generated/Early/LocalVariable.h"

class model::LocalVariable : public model::generated::LocalVariable {
public:
  using generated::LocalVariable::LocalVariable;

public:
  bool verify() const debug_function;
  bool verify(bool Assert) const debug_function;
  bool verify(VerifyHelper &VH) const;
};

#include "revng/Model/Generated/Late/LocalVariable.h"
