/// \file FlattenPrimitiveTypedefs.cpp
/// Implementation of typedef-to-primitive flattening

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Model/Binary.h"
#include "revng/Model/Pass/FlattenPrimitiveTypedefs.h"
#include "revng/Model/Pass/RegisterModelPass.h"
#include "revng/Model/Processing.h"

static RegisterModelPass R("flatten-primitive-typedefs",
                           "This looks for typedefs of common primitive types "
                           "and replaces them with their type.",
                           model::flattenPrimitiveTypedefs);

void model::flattenPrimitiveTypedefs(TupleTree<model::Binary> &Binary) {
  // Gather replacement candidates
  using TypeDefinitionKey = model::TypeDefinition::Key;
  std::map<TypeDefinitionKey, model::UpcastableType> Replacements;
  for (model::UpcastableTypeDefinition &Def : Binary->TypeDefinitions()) {
    if (auto *Typedef = llvm::dyn_cast<model::TypedefDefinition>(Def.get())) {
      if (auto &&Primitive = model::PrimitiveType::fromCName(Typedef->Name())) {
        if (Typedef->size() == Primitive->size()) {
          auto [_, Success] = Replacements.try_emplace(Typedef->key(),
                                                       std::move(Primitive));
          revng_assert(Success);
        }
      }
    }
  }

  if (Replacements.empty())
    return;

  // Replace them
  auto Visitor = [&Replacements](auto &Element) {
    using T = std::decay_t<decltype(Element)>;
    if constexpr (std::is_same_v<T, model::UpcastableType>) {
      if (Element.isEmpty())
        return;

      if (const model::TypeDefinition *Def = Element->tryGetAsDefinition())
        if (auto It = Replacements.find(Def->key()); It != Replacements.end())
          Element = model::UpcastableType(It->second);
    }
  };
  visitTupleTree(/* .Root = */ *Binary,
                 /* .PreVisitor = */ [](const auto &) {},
                 /* .PostVisitor = */ Visitor);

  // Remove leftover types
  Binary->TypeDefinitions().erase_if([&Replacements](auto &&Definition) {
    return Replacements.contains(Definition->key());
  });
}
