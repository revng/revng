#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <cstdint>
#include <optional>
#include <utility>

#include "llvm/ADT/ArrayRef.h"

#include "revng/Model/Binary.h"
#include "revng/Support/Generator.h"
#include "revng/Support/OverflowSafeInt.h"

/// Provide a view onto a raw binary through the lens of the model
class RawBinaryView {
private:
  using OverflowSafeInt = OverflowSafeInt<uint64_t>;

private:
  const model::Binary &Binary;
  llvm::ArrayRef<uint8_t> Data;

public:
  RawBinaryView(const model::Binary &Binary, llvm::StringRef Data) :
    RawBinaryView(Binary, { Data.bytes_begin(), Data.bytes_end() }) {}

  RawBinaryView(const model::Binary &Binary, llvm::ArrayRef<uint8_t> Data) :
    Binary(Binary), Data(Data) {}

public:
  uint64_t size() { return Data.size(); }

public:
  std::optional<llvm::ArrayRef<uint8_t>> getByOffset(uint64_t Offset,
                                                     uint64_t Size) const {
    auto Sum = OverflowSafeInt(Offset) + Size;
    if (not Sum or *Sum > Data.size())
      return std::nullopt;

    return Data.slice(Offset, Size);
  }

  std::optional<llvm::ArrayRef<uint8_t>> getByAddress(MetaAddress Address,
                                                      uint64_t Size) const {
    auto Offset = addressToOffset(Address);
    if (not Offset)
      return std::nullopt;

    return getByOffset(*Offset, Size);
  }

  std::optional<llvm::StringRef> getStringByAddress(MetaAddress Address,
                                                    uint64_t Size) const {
    auto BytesOrNone = getByAddress(Address, Size);
    if (not BytesOrNone.has_value())
      return std::nullopt;
    llvm::ArrayRef<uint8_t> Bytes = BytesOrNone.value();
    return llvm::StringRef(reinterpret_cast<const char *>(Bytes.data()), Size);
  }

  std::optional<uint64_t>
  readInteger(MetaAddress Address, uint64_t Size, bool IsLittleEndian) const {
    auto MaybeData = getByAddress(Address, Size);
    if (not MaybeData)
      return std::nullopt;

    llvm::support::endianness Endianness;
    if (IsLittleEndian) {
      Endianness = llvm::support::little;
    } else {
      Endianness = llvm::support::big;
    }

    switch (Size) {
    case 1:
      return (*MaybeData)[0];
    case 2:
      return llvm::support::endian::read16(MaybeData->data(), Endianness);
    case 4:
      return llvm::support::endian::read32(MaybeData->data(), Endianness);
    case 8:
      return llvm::support::endian::read64(MaybeData->data(), Endianness);
    default:
      revng_abort("Unexpected read size");
    }
  }

  std::optional<uint64_t> readInteger(MetaAddress Address,
                                      uint64_t Size) const {
    auto Architecture = Binary.Architecture();
    bool IsLittleEndian = model::Architecture::isLittleEndian(Architecture);
    return readInteger(Address, Size, IsLittleEndian);
  }

  std::optional<llvm::ArrayRef<uint8_t>>
  getFromAddressOn(MetaAddress Address) const {
    auto [Segment, OffsetInSegment] = findOffsetInSegment(Address, 0);
    if (Segment == nullptr)
      return std::nullopt;

    auto StartOffset = OverflowSafeInt(Segment->StartOffset())
                       + OffsetInSegment;
    auto Size = OverflowSafeInt(Segment->endOffset()) - StartOffset;
    if (not Size or not StartOffset)
      return std::nullopt;

    return Data.slice(*StartOffset, *Size);
  }

  /// \note This function ignores the underlying data, it just performs address
  ///       translation.
  std::optional<uint64_t> addressToOffset(MetaAddress Address,
                                          uint64_t Size = 0) const {
    auto [Segment, OffsetInSegment] = findOffsetInSegment(Address, Size);
    if (Segment == nullptr) {
      return std::nullopt;
    } else {
      auto Offset = OverflowSafeInt(Segment->StartOffset()) + OffsetInSegment;
      if (not Offset)
        return std::nullopt;

      return *Offset;
    }
  }

  /// \return the Generic address corresponding to a certain offset, if there's
  ///         one and only one.
  MetaAddress offsetToAddress(uint64_t Offset) const {
    using namespace model;

    const Segment *Match = nullptr;
    for (const Segment &Segment : Binary.Segments()) {
      if (Segment.StartOffset() <= Offset and Offset < Segment.endOffset()) {
        if (Match != nullptr) {
          // We have more than one match!
          Match = nullptr;
          break;
        }

        Match = &Segment;
      }
    }

    if (Match != nullptr) {
      auto OffsetInSegment = OverflowSafeInt(Offset) - Match->StartOffset();
      if (OffsetInSegment) {
        return Match->StartAddress() + *OffsetInSegment;
      }
    }

    return MetaAddress::invalid();
  }

  [[nodiscard]] bool isReadOnly(MetaAddress Address, uint64_t Size) const {
    for (const model::Segment &Segment : Binary.Segments()) {
      if (Segment.contains(Address, Size)) {
        if (!Segment.IsWriteable()) {
          return true;
        }
      }
    }

    return false;
  }

  using SegmentDataPair = std::pair<const model::Segment &,
                                    llvm::ArrayRef<uint8_t>>;
  cppcoro::generator<SegmentDataPair> segments() const {
    for (const model::Segment &Segment : Binary.Segments()) {
      auto MaybeData = getByOffset(Segment.StartOffset(), Segment.FileSize());
      if (MaybeData)
        co_yield SegmentDataPair(Segment, *MaybeData);
    }
  }

private:
  std::pair<const model::Segment *, uint64_t>
  findOffsetInSegment(MetaAddress Address, uint64_t Size) const {
    const model::Segment *Match = nullptr;
    for (const model::Segment &Segment : Binary.Segments()) {
      if (Segment.contains(Address, Size)) {

        if (Match != nullptr) {
          // We have more than one match!
          Match = nullptr;
          break;
        }

        Match = &Segment;
      }
    }

    if (Match != nullptr) {
      auto Offset = OverflowSafeInt(Address.address())
                    - Match->StartAddress().address();
      if (Offset)
        return { Match, *Offset };
    }

    return { nullptr, 0 };
  }
};
