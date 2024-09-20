//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Model/Binary.h"
#include "revng/Model/Filters.h"

static bool verify(const model::TypeDefinition &ModelType, const bool Assert) {
  return ModelType.verify(Assert);
}

static bool verify(const model::Type &ModelType, const bool Assert) {
  return ModelType.verify(Assert);
}

static bool verify(const model::Binary &Tree, const bool Assert) {
  return Tree.verify(Assert);
}

static TupleTree<model::Binary>
serializeDeserialize(const TupleTree<model::Binary> &T) {

  std::string Buffer;
  T.serialize(Buffer);

  auto Deserialized = TupleTree<model::Binary>::fromString(Buffer);

  std::string OtherBuffer;
  Deserialized->serialize(OtherBuffer);

  return std::move(Deserialized.get());
}

static bool checkSerialization(const TupleTree<model::Binary> &T) {
  revng_check(T->verify(true));
  auto Deserialized = serializeDeserialize(T);
  revng_check(Deserialized->verify(true));
  return T->TypeDefinitions() == Deserialized->TypeDefinitions();
}

#include "revng/tests/unit/ModelType.inc"
