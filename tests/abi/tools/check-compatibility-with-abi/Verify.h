#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <optional>
#include <string>
#include <vector>

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/YAMLTraits.h"

#include "revng/ABI/FunctionType/Layout.h"
#include "revng/Model/Binary.h"
#include "revng/Model/TypeDefinitionKind.h"

#include "ExpectedFailures.h"

/// One tested function, as reported by `--save-layouts`.
///
/// Carries both the prototype the function had in the tested model and
/// the `abi::FunctionType::Layout` we produced for it.
struct FunctionLayoutEntry {
  std::string Name;
  model::UpcastableTypeDefinition Prototype;
  abi::FunctionType::Layout Layout;
};
LLVM_YAML_IS_SEQUENCE_VECTOR(FunctionLayoutEntry)

template<>
struct llvm::yaml::MappingTraits<FunctionLayoutEntry> {
  static void mapping(IO &IO, FunctionLayoutEntry &Entry) {
    IO.mapRequired("Name", Entry.Name);
    IO.mapRequired("Prototype", Entry.Prototype);
    IO.mapRequired("Layout", Entry.Layout);
  }
};

std::vector<Failure>
verifyABI(const TupleTree<model::Binary> &Binary,
          llvm::StringRef RuntimeArtifact,
          model::ABI::Values ABI,
          std::optional<model::TypeDefinitionKind::Values> VerifyType,
          std::optional<std::vector<FunctionLayoutEntry>> &Layouts);
