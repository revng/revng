//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Model/PrimitiveType.h"
#include "revng/Model/RawBinaryView.h"
#include "revng/SegmentReferences/StringConstants.h"
#include "revng/Support/Debug.h"

static Logger Log("string-constants");

unsigned getConstCharArrayElementSize(const model::Type &Type) {
  const model::Type *Stripped = Type.skipConstAndTypedefs();

  const model::ArrayType *Array = Stripped->getArray();
  if (Array == nullptr)
    return 0;

  const model::Type &ElementType = Array->getArrayElement();
  const model::PrimitiveType *PrimitiveType = ElementType.getPrimitive();
  if (not ElementType.IsConst() or PrimitiveType == nullptr
      or PrimitiveType->PrimitiveKind() != model::PrimitiveKind::Unsigned) {
    return 0;
  }

  if (PrimitiveType->Size() != 1 and PrimitiveType->Size() != 2)
    return 0;

  return PrimitiveType->Size();
}

UnicodeCStringView readString(RawBinaryView &BinaryView,
                              const MetaAddress &Address,
                              uint64_t ByteCount,
                              unsigned CharSize) {
  auto MaybeData = BinaryView.getByAddress(Address, ByteCount);
  if (not MaybeData.has_value()) {
    revng_log(Log, "Couldn't get the data at " << Address.toString());
    return {};
  }

  for (const UnicodeCStringView &String :
       UnicodeCStringView::getPrintable(*MaybeData)) {
    if (String.charSize() != CharSize) {
      revng_log(Log, "Unexpected char size for the string");
      continue;
    }

    if (String.data().size() != MaybeData->size()) {
      revng_log(Log, "String length does not match");
      continue;
    }

    return String;
  }

  revng_log(Log, "No printable string found at " << Address.toString());
  return {};
}
