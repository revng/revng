#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Support/MetaAddress.h"

namespace llvm {
class Function;
class Use;
class User;
} // namespace llvm

namespace model {
class Binary;
class Segment;
} // namespace model

class SegmentUsesEnumerator {
public:
  enum class SegmentAccess {
    ReadOnly,
    ExecutableOnly,
    All
  };

  struct SegmentUse {
    llvm::Use *TheUse = nullptr;
    MetaAddress Address;
  };

  using UseList = llvm::SmallVector<SegmentUse>;

private:
  const model::Binary &Binary;
  SegmentAccess SegmentAccess;

public:
  SegmentUsesEnumerator(const model::Binary &Binary,
                        enum SegmentAccess SegmentAccess) :
    Binary(Binary), SegmentAccess(SegmentAccess) {}

public:
  UseList getUses(llvm::Module &M, llvm::Function *LimitTo = nullptr);

private:
  static unsigned int getOpcode(llvm::User &User);

  static bool shouldSkip(llvm::Use &TheUse);

  static std::optional<uint64_t> getAddend(llvm::Use &TheUse);
};
