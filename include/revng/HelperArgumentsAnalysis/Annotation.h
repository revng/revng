#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <optional>
#include <tuple>

#include "llvm/ADT/DenseSet.h"

namespace llvm {
class MDNode;
class LLVMContext;
class User;
class Instruction;
} // namespace llvm

namespace aua {

using OffsetAndSize = std::pair<uint64_t, uint64_t>;

class Annotation {
private:
  static constexpr const char *MetadataKind = "revng.csua";

public:
  using OffsetAndSizeSet = llvm::DenseSet<OffsetAndSize>;

public:
  bool Escapes = false;
  OffsetAndSizeSet Reads;
  OffsetAndSizeSet Writes;

public:
  ~Annotation() {}
  bool operator==(const Annotation &) const = default;

public:
  llvm::MDNode &serializeToMetadata(llvm::LLVMContext &Context) const;

  void serialize(llvm::User &ToAnnotate);

public:
  static bool isAnnotated(llvm::Instruction &I);
  static std::optional<Annotation> deserialize(llvm::User &ToAnnotate);
  static Annotation deserializeFromMetadata(llvm::LLVMContext &Context,
                                            llvm::MDNode &MD);
};

} // namespace aua
