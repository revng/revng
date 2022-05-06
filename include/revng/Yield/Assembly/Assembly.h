#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <optional>
#include <type_traits>

#include "llvm/ADT/SmallVector.h"

#include "revng/ADT/KeyedObjectTraits.h"
#include "revng/ADT/SortedVector.h"
#include "revng/Support/MetaAddress.h"

namespace assembly {

struct Instruction {
public:
  using ByteContainer = llvm::SmallVector<uint8_t, 16>;

  enum class TagType {
    Invalid,
    Immediate,
    Memory,
    Mnemonic,
    MnemonicPrefix,
    MnemonicSuffix,
    Register,
    Whitespace
  };

  struct Tag {
    TagType Type;
    size_t FromIndex;
    size_t ToIndex;

    Tag(TagType Type,
        size_t FromIndex = std::numeric_limits<size_t>::min(),
        size_t ToIndex = std::numeric_limits<size_t>::max()) :
      Type(Type), FromIndex(FromIndex), ToIndex(ToIndex) {}
  };

public:
  MetaAddress Address = MetaAddress::invalid();
  ByteContainer Bytes = {};

  std::string Text;
  llvm::SmallVector<Tag, 4> Tags = {};

  std::optional<std::string> Opcode = std::nullopt;
  std::string Comment = {};
  std::string Error = {};

  bool HasDelayedSlot = false;
};

} // namespace assembly

template<>
struct KeyedObjectTraits<assembly::Instruction> {
  static MetaAddress key(const assembly::Instruction &I) { return I.Address; }
  static assembly::Instruction fromKey(const MetaAddress &Key) {
    return assembly::Instruction{ .Address = Key };
  }
};

namespace assembly {

struct BasicBlock {
  MetaAddress Address = MetaAddress::invalid();
  SortedVector<Instruction> Instructions = {};
  SortedVector<MetaAddress> Targets = {};

  bool IsAFallthroughTarget = false;
  bool CanBeMergedWithPredecessor = false;

  std::string CommentIndicator = ";";
  std::string LabelIndicator = ":";
};

} // namespace assembly

template<>
struct KeyedObjectTraits<assembly::BasicBlock> {
  static MetaAddress key(const assembly::BasicBlock &I) { return I.Address; }
  static assembly::BasicBlock fromKey(const MetaAddress &Key) {
    return assembly::BasicBlock{ .Address = Key };
  }
};

namespace assembly {

struct Function {
  MetaAddress Address = MetaAddress::invalid();
  SortedVector<BasicBlock> BasicBlocks = {};
};

} // namespace assembly

template<>
struct KeyedObjectTraits<assembly::Function> {
  static MetaAddress key(const assembly::Function &I) { return I.Address; }
  static assembly::Function fromKey(const MetaAddress &Key) {
    return assembly::Function{ .Address = Key };
  }
};

namespace assembly {

using Functions = SortedVector<Function>;

} // namespace assembly
