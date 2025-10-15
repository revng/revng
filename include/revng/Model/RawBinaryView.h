#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <cstdint>
#include <optional>
#include <utility>

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"

#include "revng/BasicAnalyses/MaterializedValue.h"
#include "revng/Model/Architecture.h"
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
                                                     uint64_t Size) const;

  std::optional<llvm::ArrayRef<uint8_t>> getByAddress(MetaAddress Address,
                                                      uint64_t Size) const;

  std::optional<llvm::StringRef> getStringByAddress(MetaAddress Address,
                                                    uint64_t Size) const;

  std::optional<uint64_t>
  readInteger(MetaAddress Address, uint64_t Size, bool IsLittleEndian) const;

  std::optional<uint64_t> readInteger(MetaAddress Address, uint64_t Size) const;

  std::optional<llvm::ArrayRef<uint8_t>>
  getFromAddressOn(MetaAddress Address) const;

  /// \note This function ignores the underlying data, it just performs address
  ///       translation.
  std::optional<uint64_t> addressToOffset(MetaAddress Address,
                                          uint64_t Size = 0) const;

  /// \return the Generic address corresponding to a certain offset, if there's
  ///         one and only one.
  MetaAddress offsetToAddress(uint64_t Offset) const;

  [[nodiscard]] bool isReadOnly(MetaAddress Address, uint64_t Size) const;

  using SegmentDataPair = std::pair<const model::Segment &,
                                    llvm::ArrayRef<uint8_t>>;
  cppcoro::generator<SegmentDataPair> segments() const;

public:
  MaterializedValue load(const MetaAddress &Address,
                         unsigned LoadSize,
                         bool IsLittleEndian) const;

  // Factored-out checkPrecondition, should be added to each pipe that uses
  // this class. Eventually this will be moved into Binary::verify.
  static llvm::Error checkPrecondition(const model::Binary &Binary);

private:
  std::pair<const model::Segment *, uint64_t>
  findOffsetInSegment(MetaAddress Address, uint64_t Size) const;
};
