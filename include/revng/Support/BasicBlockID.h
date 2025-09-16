#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <cstdint>
#include <string>

#include "revng/Support/MetaAddress.h"

class BasicBlockID {
private:
  MetaAddress Start = MetaAddress::invalid();
  uint64_t InliningIndex = 0;

public:
  explicit BasicBlockID() = default;
  explicit BasicBlockID(const MetaAddress &Start) : Start(Start) {}
  explicit BasicBlockID(const MetaAddress &Start, uint64_t Index) :
    Start(Start), InliningIndex(Index) {
    revng_assert(verify());
  }
  /// Create an invalid BasicBlockID
  static constexpr BasicBlockID invalid() { return BasicBlockID(); }

public:
  const MetaAddress &start() const { return Start; }
  uint64_t inliningIndex() const { return InliningIndex; }
  const MetaAddress &notInlinedAddress() const {
    revng_assert(InliningIndex == 0);
    return start();
  }

public:
  [[nodiscard]] bool isInlined() const { return InliningIndex != 0; }
  [[nodiscard]] bool isValid() const { return Start.isValid(); }

public:
  static BasicBlockID fromString(llvm::StringRef Text);
  std::string toString(model::Architecture::Values Architecture =
                         model::Architecture::Invalid) const;

  static BasicBlockID fromValue(llvm::Value *V);
  llvm::Constant *toValue(llvm::Module *M) const;

public:
  /// @{
  constexpr bool operator==(const BasicBlockID &Other) const {
    return tie() == Other.tie();
  }

  constexpr bool operator!=(const BasicBlockID &Other) const {
    return not(*this == Other);
  }

  constexpr bool operator<(const BasicBlockID &Other) const {
    return tie() < Other.tie();
  }
  constexpr bool operator<=(const BasicBlockID &Other) const {
    return tie() <= Other.tie();
  }
  constexpr bool operator>(const BasicBlockID &Other) const {
    return tie() > Other.tie();
  }
  constexpr bool operator>=(const BasicBlockID &Other) const {
    return tie() >= Other.tie();
  }
  constexpr std::strong_ordering operator<=>(const BasicBlockID &Other) const {
    return tie() <=> Other.tie();
  }

  /// @}

public:
  [[nodiscard]] bool verify() const {
    if (Start.isInvalid()) {
      return InliningIndex == 0;
    }

    return true;
  }

private:
  using Tied = std::tuple<const MetaAddress &, const uint64_t &>;
  constexpr Tied tie() const { return std::tie(Start, InliningIndex); }
};

template<>
struct KeyedObjectTraits<BasicBlockID>
  : public IdentityKeyedObjectTraits<BasicBlockID> {};
