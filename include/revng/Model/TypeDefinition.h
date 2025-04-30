#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <optional>

#include "revng/ADT/RecursiveCoroutine.h"
#include "revng/ADT/SortedVector.h"
#include "revng/ADT/UpcastablePointer.h"
#include "revng/ADT/UpcastablePointer/YAMLTraits.h"
#include "revng/Model/ABI.h"
#include "revng/Model/CommonTypeMethods.h"
#include "revng/Model/Register.h"
#include "revng/Model/TypeDefinitionKind.h"
#include "revng/Support/Assert.h"
#include "revng/Support/Debug.h"
#include "revng/Support/YAMLTraits.h"
#include "revng/TupleTree/TupleTree.h"

#include "revng/Model/Generated/Early/TypeDefinition.h"

class model::TypeDefinition : public model::generated::TypeDefinition,
                              public model::CommonTypeMethods<TypeDefinition> {
public:
  using generated::TypeDefinition::TypeDefinition;

  llvm::SmallVector<const model::Type *, 4> edges() const;
  llvm::SmallVector<model::Type *, 4> edges();
  void dumpTypeGraph(const char *Path,
                     const model::Binary &Binary) const debug_function;

public:
  bool verify() const debug_function;
  bool verify(bool Assert) const debug_function;
  RecursiveCoroutine<bool> verify(VerifyHelper &VH) const;

public:
  /// Tries to recursively compute the size of the type.
  ///
  /// Use this method only on (temporarily) invalid models, for example within
  /// the binary importer. If your model is valid, use \ref size() instead.
  ///
  /// \returns * `std::nullopt` if the size cannot be computed, for example,
  ///            when the type system loops and the type's size depends on
  ///            the type itself,
  ///          * 0 for types without the size (`void` and function types),
  ///          * size in bytes in all other cases.
  ///
  /// @{
  std::optional<uint64_t> trySize() const debug_function;
  RecursiveCoroutine<std::optional<uint64_t>> trySize(VerifyHelper &VH) const;
  /// @}
};

namespace model {
using DefinitionReference = TupleTreeReference<model::TypeDefinition,
                                               model::Binary>;
}

extern template model::DefinitionReference
model::DefinitionReference::fromString<model::Binary>(model::Binary *Root,
                                                      llvm::StringRef Path);

extern template model::DefinitionReference
model::DefinitionReference::fromString<const model::Binary>(const model::Binary
                                                              *,
                                                            llvm::StringRef);

#include "revng/Model/Generated/Late/TypeDefinition.h"
