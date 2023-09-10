#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Model/Binary.h"
#include "revng/Model/QualifiedType.h"
#include "revng/Model/Section.h"
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

inline bool checkForOverlap(const model::StructType &Struct,
                            const model::StructField &Field) {
  uint64_t Size = *Field.Type().size();
  for (auto &Current : Struct.Fields()) {
    uint64_t CurrentSize = *Current.Type().size();
    if ((Current.Offset() < Field.Offset() + Size))
      if (Current.Offset() + CurrentSize > Field.Offset())
        return true;
  }

  return false;
}

inline void importSymbolsInto(model::Binary &Binary,
                              llvm::SmallVector<DataSymbol, 32> &DataSymbols,
                              model::StructType *Struct,
                              MetaAddress StructStartAddress) {
  MetaAddress StructEndAddress = StructStartAddress + Struct->Size();
  if (not StructEndAddress.isValid())
    return;

  for (auto It = DataSymbols.begin(); It != DataSymbols.end();) {
    const auto &[SymbolStartAddress, SymbolSize, SymbolName] = *It;
    MetaAddress SymbolEndAddress = SymbolStartAddress + SymbolSize;

    bool Consume = false;
    if (SymbolEndAddress.isValid()
        and SymbolStartAddress.addressGreaterThanOrEqual(StructStartAddress)
        and SymbolEndAddress.addressLowerThanOrEqual(StructEndAddress)) {
      auto Offset = SymbolStartAddress - StructStartAddress;
      revng_assert(Offset.has_value());

      model::QualifiedType FieldType;
      model::TypePath T;

      if (SymbolSize == 1 || SymbolSize == 2 || SymbolSize == 4
          || SymbolSize == 8) {
        T = Binary.getPrimitiveType(model::PrimitiveTypeKind::Generic,
                                    SymbolSize);
        FieldType = { T, {} };
      } else {
        // Replace non-standardly sized types with an array of bytes
        T = Binary.getPrimitiveType(model::PrimitiveTypeKind::Generic, 1);
        FieldType = { T, { { model::QualifierKind::Array, SymbolSize } } };
      }

      model::StructField Field{ *Offset, {}, SymbolName.str(), {}, FieldType };
      if (!checkForOverlap(*Struct, Field)) {
        // Discard any symbols that would overlap an already existing one.
        const auto &[_, Success] = Struct->Fields().insert(Field);
        revng_assert(Success);
        Consume = true;
      }
    }

    if (Consume)
      It = DataSymbols.erase(It);
    else
      ++It;
  }
}

inline model::TypePath
populateSegmentTypeStruct(model::Binary &Binary,
                          model::Segment &Segment,
                          llvm::SmallVector<DataSymbol, 32> DataSymbols) {
  using namespace llvm;
  using namespace model;

  // Create a struct for the segment
  TypePath SegmentStructPath = createEmptyStruct(Binary, Segment.VirtualSize());
  auto *SegmentStruct = cast<model::StructType>(SegmentStructPath.get());

  for (model::Section &Section : Segment.Sections()) {
    auto Offset = Section.StartAddress() - Segment.StartAddress();
    revng_assert(Offset.has_value());

    // Create a struct for each section
    TypePath SectionStructPath = createEmptyStruct(Binary, Section.Size());
    auto *SectionStruct = cast<model::StructType>(SectionStructPath.get());

    // Import (and consume) symbols that fall within such section
    importSymbolsInto(Binary,
                      DataSymbols,
                      SectionStruct,
                      Section.StartAddress());

    // Insert the field the segment struct
    StructField SectionField{
      *Offset, {}, Section.Name(), {}, { SectionStructPath, {} }
    };
    SegmentStruct->Fields().insert(SectionField);
  }

  // Pour the remaining symbols into the segment struct
  importSymbolsInto(Binary, DataSymbols, SegmentStruct, Segment.StartAddress());

  revng_assert(SegmentStruct->verify());
  return SegmentStructPath;
}
