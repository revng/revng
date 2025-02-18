#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Model/Binary.h"
#include "revng/Support/ManagedStaticRegistry.h"

struct RegisteredModelPass {
public:
  using ModelPassType = std::function<void(TupleTree<model::Binary> &)>;

public:
  std::string Name;
  std::string Description;
  ModelPassType Pass;

public:
  const std::string &key() const { return Name; }
};
using RegisterModelPass = RegisterManagedStaticImpl<RegisteredModelPass>;
