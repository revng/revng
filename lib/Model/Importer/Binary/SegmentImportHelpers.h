#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Model/Binary.h"
#include "revng/Support/Assert.h"
#include "revng/Support/IRHelpers.h"
#include "revng/Support/MetaAddress.h"

struct DataSymbol {
  MetaAddress Address;
  uint64_t Size;
  llvm::StringRef Name;

  DataSymbol(MetaAddress Address, uint64_t Size, llvm::StringRef Name) :
    Address(Address), Size(Size), Name(Name) {}

  bool operator==(const DataSymbol &) const = default;
};

using DataSymbolVecTy = llvm::SmallVectorImpl<DataSymbol>;

inline bool checkForOverlap(const model::StructType &Struct,
                            const model::StructField &Field) {
  std::uint64_t Size = *Field.Type().size();
  for (auto &Current : Struct.Fields()) {
    std::uint64_t CurrentSize = *Current.Type().size();
    if ((Current.Offset() < Field.Offset() + Size))
      if (Current.Offset() + CurrentSize > Field.Offset())
        return true;
  }

  return false;
}

inline model::QualifiedType
populateSegmentTypeStruct(model::Binary &Binary,
                          model::Segment &Segment,
                          DataSymbolVecTy &DataSymbols) {
  model::TypePath StructPath = createEmptyStruct(Binary, Segment.VirtualSize());
  auto *Struct = llvm::cast<model::StructType>(StructPath.get());

  for (const auto &DataSymbol : DataSymbols) {
    const auto &[Address, Size, Name] = DataSymbol;

    if (Segment.contains(Address)) {
      auto Offset = Address - Segment.StartAddress();
      revng_assert(Offset.has_value());

      model::QualifiedType FieldType;
      model::TypePath T;

      if (Size == 1 || Size == 2 || Size == 4 || Size == 8) {
        T = Binary.getPrimitiveType(model::PrimitiveTypeKind::Generic, Size);
        FieldType = { T, {} };
      } else {
        // Replace non-standardly sized types with an array of bytes
        T = Binary.getPrimitiveType(model::PrimitiveTypeKind::Generic, 1);
        FieldType = { T, { { model::QualifierKind::Array, Size } } };
      }

      model::StructField Field{ *Offset, {}, Name.str(), {}, FieldType };
      if (!checkForOverlap(*Struct, Field)) {
        // Discard any symbols that would overlap an already existing one.
        const auto &[_, Success] = Struct->Fields().insert(Field);
        revng_assert(Success);
      }
    }
  }

  revng_assert(Struct->verify());
  return model::QualifiedType(std::move(StructPath), {});
}
