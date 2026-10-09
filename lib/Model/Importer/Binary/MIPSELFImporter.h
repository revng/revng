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
                  const ELFBinary &TheBinary,
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

  void registerMIPSRelocations() {
    using Elf_Addr = const typename llvm::object::ELFFile<T>::Elf_Addr;
    using Elf_Sym = llvm::object::Elf_Sym_Impl<T>;

    auto &GOT = this->GotPortion;
    if (not GOT->isAvailable())
      return;

    revng_log(ELFImporterLog, "Parsing relocations in MIPS .got");
    LoggerIndent Indent(ELFImporterLog);

    uint32_t Index = 0;

    // Perform local relocations on GOT.
    if (MIPSLocalGotEntries) {

      if (*MIPSLocalGotEntries > (GOT->size() / sizeof(Elf_Addr))) {
        revng_log(ELFImporterLog, "Too many GOT entries");
        *MIPSLocalGotEntries = 0;
      }

      // GOT[0] is reserved for the lazy resolver and, if its most significant
      // bit is set, GOT[1] for a pointer to the module (a GNU extension):
      // neither holds an address within the binary
      uint64_t ReservedEntries = 1;
      auto Entries = GOT->template extractAs<Elf_Addr>();
      constexpr uint64_t ModuleMask = uint64_t(1) << (8 * sizeof(Elf_Addr) - 1);
      if (Entries.size() > 1 and (Entries[1] & ModuleMask) != 0)
        ReservedEntries = 2;

      for (; Index < *MIPSLocalGotEntries; Index++) {
        if (Index < ReservedEntries)
          continue;

        auto RelocationAddress = GOT->template addressAtIndex<Elf_Addr>(Index);

        if (not RelocationAddress.isValid()) {
          revng_log(ELFImporterLog,
                    "Local GOT entry " << Index << " is not valid");
          continue;
        }

        this->registerInPlaceRelocation(RelocationAddress, sizeof(Elf_Addr));
      }
    }

    // Relocate the remaining entries of the GOT with global symbols.
    if (MIPSFirstGotSymbol and this->SymbolsCount
        and this->DynstrPortion->isAvailable()
        and this->DynsymPortion->isAvailable()) {
      auto Symbols = this->DynsymPortion->template extractAs<Elf_Sym>();
      llvm::StringRef SymbolNames = this->DynstrPortion->extractString();
      for (uint32_t SymbolIndex = *MIPSFirstGotSymbol;
           SymbolIndex < *this->SymbolsCount;
           SymbolIndex++, Index++) {
        auto RelocationAddress = GOT->template addressAtIndex<Elf_Addr>(Index);
        this->registerRelocation(RelocationAddress,
                                 llvm::ELF::R_MIPS_JUMP_SLOT,
                                 SymbolIndex,
                                 0,
                                 Symbols,
                                 SymbolNames);
      }
    }
  }

  llvm::Error import(const ImporterOptions &Options) override {
    if (llvm::Error E = ELFImporter<T, HasAddend>::import(Options))
      return E;
    registerMIPSRelocations();
    return llvm::Error::success();
  }
};
