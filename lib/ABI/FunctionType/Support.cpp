/// \file Support.cpp
/// \brief

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/ABI/FunctionType/Support.h"

namespace abi::FunctionType {

void replaceReferences(const model::Type::Key &OldKey,
                       const model::TypePath &NewTypePath,
                       TupleTree<model::Binary> &Model) {
  auto Visitor = [&](model::TypePath &Visited) {
    if (!Visited.isValid())
      return; // Ignore empty references

    model::Type *Current = Visited.get();
    revng_assert(Current != nullptr);

    if (Current->key() == OldKey)
      Visited = NewTypePath;
  };
  Model.visitReferences(Visitor);
  Model->Types().erase(OldKey);
}

} // namespace abi::FunctionType
