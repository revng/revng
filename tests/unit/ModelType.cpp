//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Model/Binary.h"

static bool verify(const model::Type &ModelType, const bool Assert) {
  return ModelType.verify(Assert);
}

static bool verify(const model::QualifiedType &ModelType, const bool Assert) {
  return ModelType.verify(Assert);
}

static bool verify(const model::Binary &Tree, const bool Assert) {
  return Tree.verify(Assert);
}

static bool checkSerialization(const TupleTree<model::Binary> &Tree) {
  return true;
}

#include "revng/tests/unit/ModelType.inc"
