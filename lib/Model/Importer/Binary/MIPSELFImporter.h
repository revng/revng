#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "ELFImporter.h"

template<typename T, bool HasAddend>
class MIPSELFImporter : public ELFImporter<T, HasAddend> {
public:
  template<typename Q>
  using SmallVectorImpl = llvm::SmallVectorImpl<Q>;

private:
  std::optional<uint64_t> MIPSFirstGotSymbol;
  std::optional<uint64_t> MIPSLocalGotEntries;

public:
  MIPSELFImporter(TupleTree<model::Binary> &Model,
                  const llvm::object::ELFObjectFileBase &TheBinary,
                  uint64_t BaseAddress) :
    ELFImporter<T, HasAddend>(Model, TheBinary, BaseAddress),
    MIPSFirstGotSymbol(std::nullopt),
    MIPSLocalGotEntries(std::nullopt) {}

  void parseTargetDynamicTags(uint64_t Tag,
                              MetaAddress Relocated,
                              SmallVectorImpl<uint64_t> &LibrariesOffsets,
                              uint64_t Val) override {
    using namespace llvm;

    switch (Tag) {
    case ELF::DT_MIPS_SYMTABNO:
      this->SymbolsCount = Val;
      break;

    case ELF::DT_MIPS_GOTSYM:
      MIPSFirstGotSymbol = Val;
      break;

    case ELF::DT_MIPS_LOCAL_GOTNO:
      MIPSLocalGotEntries = Val;
      break;
      // TODO:
      // ```
      //   case ELF::DT_PLTGOT:
      //   TODO: record canonical value of the global pointer
      //         to Relocated + 0x7ff0
      // ```
    }

    fixupMIPSGOT();
  }

  void fixupMIPSGOT() {
    using Elf_Addr = const typename llvm::object::ELFFile<T>::Elf_Addr;
    // In MIPS the GOT has one entry per symbol.
    if (this->SymbolsCount and MIPSFirstGotSymbol and MIPSLocalGotEntries) {
      uint32_t GotEntries = (*MIPSLocalGotEntries
                             + (*this->SymbolsCount - *MIPSFirstGotSymbol));
      this->GotPortion->setSize(GotEntries * sizeof(Elf_Addr));
    }
  }

  using Elf_Rel = llvm::object::Elf_Rel_Impl<T, HasAddend>;
  using Elf_Rel_Array = llvm::ArrayRef<Elf_Rel>;

  void registerMIPSRelocations() {
    using Elf_Addr = const typename llvm::object::ELFFile<T>::Elf_Addr;
    using Elf_Rel = llvm::object::Elf_Rel_Impl<T, HasAddend>;
    std::vector<Elf_Rel> MIPSImplicitRelocations;

    auto &GOT = this->GotPortion;
    uint32_t Index = 0;

    // Perform local relocations on GOT.
    if (MIPSLocalGotEntries) {

      if (*MIPSLocalGotEntries > (GOT->size() / sizeof(Elf_Addr))) {
        revng_log(ELFImporterLog, "Too many GOT entries");
        *MIPSLocalGotEntries = 0;
      }

      for (; Index < *MIPSLocalGotEntries; Index++) {
        auto RelocationAddress = GOT->template addressAtIndex<Elf_Addr>(Index);

        if (not RelocationAddress.isValid()) {
          revng_log(ELFImporterLog,
                    "Local GOT entry " << Index << " is not valid");
          continue;
        }

        Elf_Rel NewRelocation;
        NewRelocation.r_offset = RelocationAddress.address();
        NewRelocation.setSymbolAndType(0, R_MIPS_IMPLICIT_RELATIVE, false);

        MIPSImplicitRelocations.push_back(NewRelocation);
      }
    }

    // Relocate the remaining entries of the GOT with global symbols.
    if (MIPSFirstGotSymbol and this->SymbolsCount
        and this->DynstrPortion->isAvailable()
        and this->DynsymPortion->isAvailable()) {
      for (uint32_t SymbolIndex = *MIPSFirstGotSymbol;
           SymbolIndex < *this->SymbolsCount;
           SymbolIndex++, Index++) {
        auto RelocationAddress = GOT->template addressAtIndex<Elf_Addr>(Index);

        Elf_Rel NewRelocation;
        NewRelocation.r_offset = RelocationAddress.address();
        NewRelocation.setSymbolAndType(SymbolIndex,
                                       llvm::ELF::R_MIPS_JUMP_SLOT,
                                       false);
        MIPSImplicitRelocations.push_back(NewRelocation);
      }
    }

    if (GOT->isAvailable())
      this->registerRelocations(MIPSImplicitRelocations,
                                *this->DynsymPortion.get(),
                                *this->DynstrPortion.get());
  }

  llvm::Error import(const ImporterOptions &Options) override {
    if (llvm::Error E = ELFImporter<T, HasAddend>::import(Options))
      return E;
    registerMIPSRelocations();
    return llvm::Error::success();
  }
};
