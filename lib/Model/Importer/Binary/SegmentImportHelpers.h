#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Model/Binary.h"
#include "revng/Model/IRHelpers.h"
#include "revng/Model/QualifiedType.h"
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

inline bool checkForOverlap(const model::StructDefinition &Struct,
                            uint64_t Offset,
                            uint64_t Size) {
  for (auto &Current : Struct.Fields()) {
    uint64_t CurrentSize = *Current.Type().size();
    if ((Current.Offset() < Offset + Size))
      if (Current.Offset() + CurrentSize > Offset)
        return true;
  }

  return false;
}

inline void importSymbolsInto(model::Binary &Binary,
                              llvm::SmallVector<DataSymbol, 32> &DataSymbols,
                              model::StructDefinition *Struct,
                              MetaAddress StructStartAddress) {
  MetaAddress StructEndAddress = StructStartAddress + Struct->Size();
  if (not StructEndAddress.isValid())
    return;

  for (auto It = DataSymbols.begin(); It != DataSymbols.end(); ++It) {
    const auto &[SymbolStartAddress, SymbolSize, SymbolName] = *It;
    MetaAddress SymbolEndAddress = SymbolStartAddress + SymbolSize;

    if (SymbolEndAddress.isValid()
        and SymbolStartAddress.addressGreaterThanOrEqual(StructStartAddress)
        and SymbolEndAddress.addressLowerThanOrEqual(StructEndAddress)) {
      auto Offset = SymbolStartAddress - StructStartAddress;
      revng_assert(Offset.has_value());

      // Discard any symbols that would overlap an already existing one.
      if (checkForOverlap(*Struct, *Offset, SymbolSize)) {
        It = std::prev(DataSymbols.erase(It));
        continue;
      }

      model::DefinitionReference T;
      if (SymbolSize == 1 || SymbolSize == 2 || SymbolSize == 4
          || SymbolSize == 8) {
        T = Binary.getPrimitiveType(model::PrimitiveKind::Generic, SymbolSize);
      } else {
        // If the symbol has a non-standard size, make it an empty struct with
        // the same size instead.
        T = createEmptyStruct(Binary, SymbolSize);
      }

      model::QualifiedType FieldType = { T, {} };
      model::StructField Field{ *Offset, {}, SymbolName.str(), {}, FieldType };
      const auto &[_, Success] = Struct->Fields().insert(Field);
      revng_assert(Success);
    }
  }
}

struct Section {
  MetaAddress StartAddress;
  uint64_t Size;
  bool CanContainCode = false;
  std::string Name;
};

inline model::DefinitionReference
populateSegmentTypeStruct(model::Binary &Binary,
                          model::Segment &Segment,
                          llvm::SmallVector<DataSymbol, 32> DataSymbols,
                          llvm::ArrayRef<Section> Sections,
                          bool SegmentIsExecutable) {
  using namespace llvm;
  using namespace model;

  // Create a struct for the segment
  auto SegmentStructPath = createEmptyStruct(Binary, Segment.VirtualSize());
  auto *SegmentStruct = cast<model::StructDefinition>(SegmentStructPath.get());
  SegmentStruct->CanContainCode() = SegmentIsExecutable;

  for (const auto &Section : Sections) {
    if (not Segment.contains(Section.StartAddress, Section.Size))
      continue;

    auto Offset = Section.StartAddress - Segment.StartAddress();
    revng_assert(Offset.has_value());

    if (checkForOverlap(*SegmentStruct, *Offset, Section.Size))
      continue;

    // Create a struct for each section
    auto SectionStructRef = createEmptyStruct(Binary, Section.Size);
    auto *SectionStruct = cast<model::StructDefinition>(SectionStructRef.get());
    SectionStruct->CanContainCode() = (SegmentIsExecutable
                                       and Section.CanContainCode);

    // Import (and consume) symbols that fall within such section
    importSymbolsInto(Binary, DataSymbols, SectionStruct, Section.StartAddress);

    // Insert the field the segment struct
    StructField SectionField{
      *Offset, {}, Section.Name, {}, { SectionStructRef, {} }
    };
    SegmentStruct->Fields().insert(SectionField);
  }

  // Pour the remaining symbols into the segment struct
  importSymbolsInto(Binary, DataSymbols, SegmentStruct, Segment.StartAddress());

  SegmentStruct->verify(true);
  return SegmentStructPath;
}
