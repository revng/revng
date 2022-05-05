/// \file ELF.cpp
/// \brief

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <cstdint>
#include <optional>

#include "llvm/ADT/DenseMap.h"
#include "llvm/Object/ELF.h"
#include "llvm/Object/ELFObjectFile.h"
#include "llvm/Object/ObjectFile.h"

#include "revng/ABI/DefaultFunctionPrototype.h"
#include "revng/Model/Binary.h"
#include "revng/Model/Importer/Dwarf/DwarfImporter.h"
#include "revng/Model/RawBinaryView.h"
#include "revng/Support/Debug.h"

#include "BinaryImporterHelper.h"
#include "DwarfReader.h"
#include "Importers.h"

using namespace llvm;

static Logger<> Log("elf-importer");

static bool shouldIgnoreSymbol(StringRef Name) {
  return Name == "$a" or Name == "$d";
}

static bool endsWith(StringRef String, char Last) {
  return not String.empty() and String.back() == Last;
}

static llvm::StringRef
extractNullTerminatedStringAt(llvm::StringRef Source, uint64_t Offset) {
  auto Size = Source.slice(Offset, Source.size()).find('\0');
  return Source.slice(Offset, Offset + Size);
}

template<typename T>
static void logAddress(T &Logger, const char *Name, MetaAddress Address) {
  if (Logger.isEnabled()) {
    Logger << Name;
    Address.dump(Logger);
    Logger << DoLog;
  }
}

class FilePortion2 {
private:
  const RawBinaryView &File;
  bool HasAddress;
  bool HasSize;
  uint64_t Size;
  MetaAddress Address;

public:
  FilePortion2(const RawBinaryView &File) :
    File(File),
    HasAddress(false),
    HasSize(false),
    Size(0),
    Address(MetaAddress::invalid()) {}

public:
  void setAddress(MetaAddress Address) {
    HasAddress = true;
    this->Address = Address;
  }

  void setSize(uint64_t Size) {
    HasSize = true;
    this->Size = Size;
  }

  MetaAddress addressAtOffset(uint64_t Offset) {
    if (not(HasAddress and HasSize and Offset <= Size))
      return MetaAddress::invalid();
    return Address + Offset;
  }

  template<typename T>
  MetaAddress addressAtIndex(uint64_t Index) {
    uint64_t Offset = Index * sizeof(T);
    if (not(HasAddress and HasSize and Offset <= Size))
      return MetaAddress::invalid();
    return Address + Offset;
  }

  bool isAvailable() const { return HasAddress; }

  bool isExact() const {
    revng_assert(HasAddress);
    return HasSize;
  }

  StringRef extractString() const {
    auto Data = extractData();

    const char *AsChar = reinterpret_cast<const char *>(Data.data());
    return StringRef(AsChar, Data.size());
  }

  template<typename T>
  ArrayRef<T> extractAs() const {
    auto Data = extractData();

    const size_t TypeSize = sizeof(T);
    if (Data.size() % TypeSize != 0)
      return {};

    return ArrayRef<T>(reinterpret_cast<const T *>(Data.data()),
                       Data.size() / TypeSize);
  }

  ArrayRef<uint8_t> extractData() const {
    revng_assert(HasAddress);

    if (HasSize) {
      auto MaybeData = File.getByAddress(Address, Size);
      if (MaybeData) {
        return *MaybeData;
      } else {
        revng_log(Log,
                  "Cannot access address " << Address.toString() << " and size "
                                           << Size);
        return {};
      }
    } else {
      auto MaybeData = File.getFromAddressOn(Address);
      if (MaybeData) {
        return *MaybeData;
      } else {
        revng_log(Log, "Cannot access address " << Address.toString());
        return {};
      }
    }
  }
};

template<typename T, bool HasAddend>
uint64_t symbolsCount(const FilePortion2 &Relocations) {
  using Elf_Rel = llvm::object::Elf_Rel_Impl<T, HasAddend>;

  if (not Relocations.isAvailable() or not Relocations.isExact())
    return 0;

  uint32_t SymbolsCount = 0;

  for (Elf_Rel Relocation : Relocations.extractAs<Elf_Rel>())
    SymbolsCount = std::max(SymbolsCount, Relocation.getSymbol(false) + 1);

  return SymbolsCount;
}

class ELFImporter : public BinaryImporterHelper {
private:
  RawBinaryView File;
  TupleTree<model::Binary> &Model;
  const object::ELFObjectFileBase &TheBinary;
  uint64_t PreferredBaseAddress;

  Optional<MetaAddress> EHFrameHdrAddress;
  Optional<MetaAddress> DynamicAddress;

public:
  ELFImporter(TupleTree<model::Binary> &Model,
              const object::ELFObjectFileBase &TheBinary,
              uint64_t PreferredBaseAddress) :
    File(*Model, toArrayRef(TheBinary.getData())),
    Model(Model),
    TheBinary(TheBinary),
    PreferredBaseAddress(PreferredBaseAddress) {}

private:
  template<typename T, bool Addend>
  using Elf_Rel_Array = llvm::ArrayRef<llvm::object::Elf_Rel_Impl<T, Addend>>;

  template<typename T>
  using ConstElf_Shdr = const typename object::ELFFile<T>::Elf_Shdr;

public:
  template<typename T, bool HasAddend>
  llvm::Error import();

private:
  template<typename T>
  MetaAddress getGenericPointer(Pointer Ptr) const {
    if (not Ptr.isIndirect())
      return Ptr.value();

    auto MaybePointer = File.getFromAddressOn(Ptr.value());
    if (not MaybePointer)
      return MetaAddress::invalid();

    return fromGeneric(::readPointer<T>(MaybePointer->data()));
  }

  template<typename T>
  MetaAddress getCodePointer(Pointer Ptr) const {
    using namespace model::Architecture;
    auto Architecture = Model->Architecture;
    return getGenericPointer<T>(Ptr).toPC(toLLVMArchitecture(Architecture));
  }

  /// \brief Parse the .eh_frame_hdr section to obtain the address and the
  ///        number of FDEs in .eh_frame
  ///
  /// \return a pair containing a (possibly invalid) pointer to the .eh_frame
  ///         section and the count of FDEs in the .eh_frame_hdr section (which
  ///         should match the number of FDEs in .eh_frame)
  template<typename T>
  std::pair<MetaAddress, uint64_t> ehFrameFromEhFrameHdr();

  /// \brief Parse the .eh_frame section to collect all the landing pads
  ///
  /// \param EHFrameAddress the address of the .eh_frame section
  /// \param FDEsCount the count of FDEs in the .eh_frame section
  /// \param EHFrameSize the size of the .eh_frame section
  ///
  /// \note Either \p FDEsCount or \p EHFrameSize have to be specified
  template<typename T>
  void parseEHFrame(MetaAddress EHFrameAddress,
                    Optional<uint64_t> FDEsCount,
                    Optional<uint64_t> EHFrameSize);

  /// \brief Parse an LSDA to collect its landing pads
  ///
  /// \param FDEStart the start address of the FDE to which this LSDA is
  ///        associated
  /// \param LSDAAddress the address of the target LSDA
  template<typename T>
  void parseLSDA(MetaAddress FDEStart, MetaAddress LSDAAddress);

  /// \brief Register a label for each input relocation
  template<typename T, bool HasAddend>
  void registerRelocations(Elf_Rel_Array<T, HasAddend> Relocations,
                           const FilePortion2 &Dynsym,
                           const FilePortion2 &Dynstr);

  template<typename T, bool HasAddend>
  void
  parseSymbols(object::ELFFile<T> &TheELF, ConstElf_Shdr<T> *SectionHeader);

  template<typename T, bool HasAddend>
  void parseProgramHeaders(object::ELFFile<T> &TheELF);

  template<typename T>
  void
  parseDynamicSymbol(llvm::object::Elf_Sym_Impl<T> &Symbol, StringRef Dynstr);
};

template<typename T, bool HasAddend>
Error ELFImporter::import() {
  // Parse the ELF file
  auto TheELFOrErr = object::ELFFile<T>::create(TheBinary.getData());
  if (not TheELFOrErr)
    return TheELFOrErr.takeError();
  object::ELFFile<T> &TheELF = *TheELFOrErr;

  revng_assert(Model->Architecture != model::Architecture::Invalid);
  Architecture = Model->Architecture;

  // Set default ABI
  Model->DefaultABI = model::ABI::getDefault(Model->Architecture);

  // BaseAddress makes sense only for shared (relocatable, PIC) objects
  auto Type = TheELF.getHeader().e_type;
  if (Type == ELF::ET_DYN)
    BaseAddress = PreferredBaseAddress;

  if (not(Type == ELF::ET_DYN or Type == ELF::ET_EXEC))
    return createError("Only ELF executables and ELF dynamic libraries are "
                       "supported");

  // Look for static or dynamic symbols and relocations
  ConstElf_Shdr<T> *SymtabShdr = nullptr;
  Optional<MetaAddress> EHFrameAddress;
  Optional<uint64_t> EHFrameSize;

  auto Sections = TheELF.sections();
  if (not Sections) {
    logAllUnhandledErrors(std::move(Sections.takeError()), errs(), "");
  } else {
    for (auto &Section : *Sections) {
      auto NameOrErr = TheELF.getSectionName(Section);
      if (NameOrErr) {
        auto &Name = *NameOrErr;
        if (Name == ".symtab") {
          // TODO: check dedicated field in section header
          if (SymtabShdr == nullptr)
            SymtabShdr = &Section;
          else
            revng_log(Log, "Multiple .symtab. Ignoring.");
        } else if (Name == ".eh_frame") {
          if (not EHFrameAddress) {
            EHFrameAddress = relocate(fromGeneric(Section.sh_addr));
            EHFrameSize = static_cast<uint64_t>(Section.sh_size);
          } else {
            revng_log(Log, "Duplicate .eh_frame. Ignoring.");
          }
        } else if (Name == ".dynamic") {
          if (not DynamicAddress)
            DynamicAddress = relocate(fromGeneric(Section.sh_addr));
          else
            revng_log(Log, "Duplicate .dynamic. Ignoring.");
        }
      }
    }
  }

  parseSymbols<T, HasAddend>(TheELF, SymtabShdr);

  const auto &ElfHeader = TheELF.getHeader();
  Model->EntryPoint = relocate(fromPC(ElfHeader.e_entry));

  parseProgramHeaders<T, HasAddend>(TheELF);

  using Elf_Addr = const typename object::ELFFile<T>::Elf_Addr;

  Optional<uint64_t> FDEsCount;
  if (EHFrameHdrAddress) {
    MetaAddress Address = MetaAddress::invalid();

    std::tie(Address, FDEsCount) = this->ehFrameFromEhFrameHdr<T>();
    if (Address.isValid()) {
      if (EHFrameAddress and *EHFrameAddress != Address) {
        revng_log(Log,
                  "Incoherent .eh_frame information: .eh_frame is at "
                    << EHFrameAddress->toString()
                    << " while .eh_frame_hdr reports " << Address.toString());
      }

      EHFrameAddress = Address;
    }
  }

  if (EHFrameAddress and EHFrameAddress->isValid())
    parseEHFrame<T>(*EHFrameAddress, FDEsCount, EHFrameSize);

  // Parse the .dynamic table
  auto DynamicEntries = TheELF.dynamicEntries();
  if (DynamicEntries) {
    SmallVector<uint64_t, 10> NeededLibraryNameOffsets;

    FilePortion2 DynstrPortion(File);
    FilePortion2 DynsymPortion(File);
    FilePortion2 ReldynPortion(File);
    FilePortion2 RelpltPortion(File);
    FilePortion2 GotPortion(File);
    Optional<uint64_t> SymbolsCount;
    Optional<uint64_t> MIPSFirstGotSymbol;
    Optional<uint64_t> MIPSLocalGotEntries;
    bool IsMIPS = (Model->Architecture == model::Architecture::mips
                   or Model->Architecture == model::Architecture::mipsel);

    using Elf_Dyn = const typename object::ELFFile<T>::Elf_Dyn;
    for (Elf_Dyn &DynamicTag : *DynamicEntries) {

      auto TheTag = DynamicTag.getTag();
      MetaAddress Relocated = relocate(fromGeneric(DynamicTag.getPtr()));
      switch (TheTag) {
      case ELF::DT_NEEDED:
        NeededLibraryNameOffsets.push_back(DynamicTag.getVal());
        break;

      case ELF::DT_STRTAB:
        DynstrPortion.setAddress(Relocated);
        break;

      case ELF::DT_STRSZ:
        DynstrPortion.setSize(DynamicTag.getVal());
        break;

      case ELF::DT_SYMTAB:
        DynsymPortion.setAddress(Relocated);
        break;

      case ELF::DT_JMPREL:
        RelpltPortion.setAddress(Relocated);
        break;

      case ELF::DT_PLTRELSZ:
        RelpltPortion.setSize(DynamicTag.getVal());
        break;

      case ELF::DT_REL:
      case ELF::DT_RELA:
        if (TheTag != (HasAddend ? ELF::DT_RELA : ELF::DT_REL)) {
          if (TheTag == ELF::DT_RELA)
            revng_log(Log, "Unexpected addend in relocation");
          else
            revng_log(Log, "Addend was expected in relocation");
        }
        ReldynPortion.setAddress(Relocated);
        break;

      case ELF::DT_RELSZ:
      case ELF::DT_RELASZ:
        if (TheTag != (HasAddend ? ELF::DT_RELASZ : ELF::DT_RELSZ)) {
          if (TheTag == ELF::DT_RELASZ)
            revng_log(Log, "Unexpected addend in relocation");
          else
            revng_log(Log, "Addend was expected in relocation");
        }
        ReldynPortion.setSize(DynamicTag.getVal());
        break;

      case ELF::DT_PLTGOT:
        GotPortion.setAddress(Relocated);

        if (IsMIPS) {
          // TODO: record canonical value of the global pointer to Relocated +
          //       0x7ff0
        }

        break;

      case ELF::DT_MIPS_SYMTABNO:
        if (IsMIPS)
          SymbolsCount = DynamicTag.getVal();
        break;

      case ELF::DT_MIPS_GOTSYM:
        if (IsMIPS)
          MIPSFirstGotSymbol = DynamicTag.getVal();
        break;

      case ELF::DT_MIPS_LOCAL_GOTNO:
        if (IsMIPS)
          MIPSLocalGotEntries = DynamicTag.getVal();
        break;
      }
    }

    // In MIPS the GOT has one entry per symbol
    if (IsMIPS and SymbolsCount and MIPSFirstGotSymbol
        and MIPSLocalGotEntries) {
      uint32_t GotEntries = (*MIPSLocalGotEntries
                             + (*SymbolsCount - *MIPSFirstGotSymbol));
      GotPortion.setSize(GotEntries * sizeof(Elf_Addr));
    }

    StringRef Dynstr;

    if (DynstrPortion.isAvailable()) {
      Dynstr = DynstrPortion.extractString();
      auto Inserter = Model->ImportedLibraries.batch_insert();
      for (auto Offset : NeededLibraryNameOffsets) {
        StringRef LibraryName = extractNullTerminatedStringAt(Dynstr, Offset);
        revng_assert(not endsWith(LibraryName, '\0'));
        Inserter.insert(LibraryName.data());
      }
    }

    // Collect symbols count and code pointers in image base-relative
    // relocations

    if (not SymbolsCount) {
      SymbolsCount = std::max(symbolsCount<T, HasAddend>(ReldynPortion),
                              symbolsCount<T, HasAddend>(RelpltPortion));
    }

    // Collect function addresses contained in dynamic symbols
    if (SymbolsCount and *SymbolsCount > 0 and DynsymPortion.isAvailable()) {
      using Elf_Sym = llvm::object::Elf_Sym_Impl<T>;
      DynsymPortion.setSize(*SymbolsCount * sizeof(Elf_Sym));

      ArrayRef<Elf_Sym> Symbols = DynsymPortion.extractAs<Elf_Sym>();

      for (Elf_Sym Symbol : Symbols)
        parseDynamicSymbol<T>(Symbol, Dynstr);

      using Elf_Rel = llvm::object::Elf_Rel_Impl<T, HasAddend>;
      if (ReldynPortion.isAvailable()) {
        registerRelocations<T, HasAddend>(ReldynPortion.extractAs<Elf_Rel>(),
                                          DynsymPortion,
                                          DynstrPortion);
      }

      if (RelpltPortion.isAvailable()) {
        registerRelocations<T, HasAddend>(RelpltPortion.extractAs<Elf_Rel>(),
                                          DynsymPortion,
                                          DynstrPortion);
      }

      if (IsMIPS and GotPortion.isAvailable()) {
        std::vector<Elf_Rel> MIPSImplicitRelocations;
        uint32_t GotIndex = 0;

        // Perform local relocations on GOT
        if (MIPSLocalGotEntries) {
          for (; GotIndex < *MIPSLocalGotEntries; GotIndex++) {
            auto Address = GotPortion.addressAtIndex<Elf_Addr>(GotIndex);
            Elf_Rel NewRelocation;
            NewRelocation.r_offset = Address.address();
            NewRelocation.setSymbolAndType(0, R_MIPS_IMPLICIT_RELATIVE, false);
            MIPSImplicitRelocations.push_back(NewRelocation);
          }
        }

        // Relocate the remaining entries of the GOT with global symbols
        if (MIPSFirstGotSymbol and SymbolsCount and DynstrPortion.isAvailable()
            and DynsymPortion.isAvailable()) {
          for (uint32_t SymbolIndex = *MIPSFirstGotSymbol;
               SymbolIndex < *SymbolsCount;
               SymbolIndex++, GotIndex++) {
            auto Address = GotPortion.addressAtIndex<Elf_Addr>(GotIndex);

            Elf_Rel NewRelocation;
            NewRelocation.r_offset = Address.address();
            NewRelocation.setSymbolAndType(SymbolIndex,
                                           llvm::ELF::R_MIPS_JUMP_SLOT,
                                           false);
            MIPSImplicitRelocations.push_back(NewRelocation);
          }
        }

        auto Relocations = ArrayRef<Elf_Rel>(MIPSImplicitRelocations);
        registerRelocations<T, HasAddend>(Relocations,
                                          DynsymPortion,
                                          DynstrPortion);
      }
    }
  }

  // Create a default prototype
  Model->DefaultPrototype = abi::registerDefaultFunctionPrototype(*Model.get());

  // Import Dwarf
  DwarfImporter Importer(Model);
  Importer.import(TheBinary, "");

  return Error::success();
}

template<typename T, bool HasAddend>
void ELFImporter::parseSymbols(object::ELFFile<T> &TheELF,
                               ConstElf_Shdr<T> *SymtabShdr) {
  // Check if we found a symbol table
  if (SymtabShdr == nullptr or SymtabShdr->sh_link == 0)
    return;

  // Obtain a reference to the string table
  auto Strtab = TheELF.getSection(SymtabShdr->sh_link);
  if (not Strtab) {
    revng_log(Log, "Cannot find .strtab: " << Strtab.takeError());
    return;
  }

  auto StrtabArray = TheELF.getSectionContents(**Strtab);
  if (not StrtabArray) {
    revng_log(Log, "Cannot access .strtab: " << StrtabArray.takeError());
    return;
  }

  StringRef StrtabContent(reinterpret_cast<const char *>(StrtabArray->data()),
                          StrtabArray->size());

  // Collect symbol names
  auto ELFSymbols = TheELF.symbols(SymtabShdr);
  if (not ELFSymbols) {
    revng_log(Log, "Cannot get symbols: " << ELFSymbols.takeError());
    return;
  }

  for (auto &Symbol : *ELFSymbols) {
    auto MaybeName = expectedToOptional(Symbol.getName(StrtabContent));

    bool IsCode = Symbol.getType() == ELF::STT_FUNC;
    if ((MaybeName and shouldIgnoreSymbol(*MaybeName))
        or (Symbol.st_shndx == ELF::SHN_UNDEF))
      continue;

    MetaAddress Address = MetaAddress::invalid();

    if (IsCode)
      Address = relocate(fromPC(Symbol.st_value));
    else
      Address = relocate(fromGeneric(Symbol.st_value));

    if (IsCode) {
      auto It = Model->Functions.find(Address);
      if (It == Model->Functions.end()) {
        model::Function &Function = Model->Functions[Address];
        Function.Type = model::FunctionType::Regular;
        if (MaybeName)
          Function.OriginalName = *MaybeName;
      }
    }
  }
}

template<typename A, typename B>
static bool hasFlag(A Flag, B Value) {
  return (Flag & Value) != 0;
}

template<typename T, bool HasAddend>
void ELFImporter::parseProgramHeaders(object::ELFFile<T> &TheELF) {
  // Loop over the program headers looking for PT_LOAD segments, read them out
  // and create a global variable for each one of them (writable or read-only),
  // assign them a section and output information about them in the linking info
  // CSV
  using Elf_Phdr = const typename object::ELFFile<T>::Elf_Phdr;

  Elf_Phdr *DynamicPhdr = nullptr;

  auto ProgHeaders = TheELF.program_headers();
  if (not ProgHeaders) {
    revng_log(Log,
              "Cannot access program headers: " << ProgHeaders.takeError());
    return;
  }

  for (Elf_Phdr &ProgramHeader : *ProgHeaders) {
    switch (ProgramHeader.p_type) {
    case ELF::PT_LOAD: {
      auto Start = relocate(fromGeneric(ProgramHeader.p_vaddr));
      auto EndVirtualAddress = Start + u64(ProgramHeader.p_memsz);
      if (Start.isInvalid() or EndVirtualAddress.isInvalid()) {
        revng_log(Log, "Invalid segment found");
        continue;
      }

      model::Segment NewSegment({ Start, ProgramHeader.p_memsz });

      NewSegment.StartOffset = ProgramHeader.p_offset;

      auto MaybeEndOffset = (OverflowSafeInt(u64(ProgramHeader.p_offset))
                             + u64(ProgramHeader.p_filesz));
      if (not MaybeEndOffset) {
        revng_log(Log,
                  "Invalid segment found: overflow in computing end offset");
        continue;
      }
      NewSegment.FileSize = ProgramHeader.p_filesz;

      NewSegment.IsReadable = hasFlag(ProgramHeader.p_flags, ELF::PF_R);
      NewSegment.IsWriteable = hasFlag(ProgramHeader.p_flags, ELF::PF_W);
      NewSegment.IsExecutable = hasFlag(ProgramHeader.p_flags, ELF::PF_X);

      // If it's an executable segment, and we've been asked so, register
      // which sections actually contain code
      auto Sections = TheELF.sections();
      if (not Sections) {
        logAllUnhandledErrors(std::move(Sections.takeError()), errs(), "");
      } else {
        using Elf_Shdr = const typename object::ELFFile<T>::Elf_Shdr;
        auto Inserter = NewSegment.Sections.batch_insert();
        for (Elf_Shdr &SectionHeader : *Sections) {

          if (not hasFlag(SectionHeader.sh_flags, ELF::SHF_ALLOC))
            continue;

          bool ContainsCode = (NewSegment.IsExecutable
                               and hasFlag(SectionHeader.sh_flags,
                                           ELF::SHF_EXECINSTR));
          auto SectionStart = relocate(fromGeneric(SectionHeader.sh_addr));
          uint64_t Size = SectionHeader.sh_size;
          auto SectionEnd = SectionStart + Size;

          if (SectionStart.isValid() and SectionEnd.isValid()
              and SectionStart.addressLowerThan(SectionEnd)
              and NewSegment.contains(SectionStart, Size)) {
            model::Section NewSection(SectionStart, SectionHeader.sh_size);
            if (auto SectionName = TheELF.getSectionName(SectionHeader);
                SectionName)
              NewSection.Name = SectionName->str();
            NewSection.ContainsCode = ContainsCode;
            NewSection.verify(true);
            Inserter.insert(std::move(NewSection));
          }
        }
      }

      NewSegment.verify(true);

      Model->Segments.insert(std::move(NewSegment));

    } break;

    case ELF::PT_GNU_EH_FRAME:
      if (not EHFrameHdrAddress)
        EHFrameHdrAddress = relocate(fromGeneric(ProgramHeader.p_vaddr));
      else
        revng_log(Log, "Multiple PT_GNU_EH_FRAME. Ignoring.");
      break;

    case ELF::PT_DYNAMIC:
      if (DynamicPhdr != nullptr) {
        revng_log(Log, "Duplicate .dynamic program header");
        break;
      }
      DynamicPhdr = &ProgramHeader;
      MetaAddress DynamicPhdrMA = relocate(fromGeneric(DynamicPhdr->p_vaddr));

      if (DynamicAddress and DynamicPhdrMA != *DynamicAddress) {
        revng_log(Log,
                  "Different addresses for .dynamic ("
                    << DynamicAddress->toString()
                    << ") and PT_DYNAMIC program header ("
                    << DynamicPhdrMA.toString() << ")");
        break;
      }

      DynamicAddress = relocate(DynamicPhdrMA);

      break;
    }
  }

  if ((DynamicPhdr != nullptr) != (DynamicAddress.hasValue())) {
    revng_log(Log, "Invalid .dynamic/PT_DYNAMIC");
    DynamicPhdr = nullptr;
    DynamicAddress = {};
  }
}

template<typename T>
void ELFImporter::parseDynamicSymbol(llvm::object::Elf_Sym_Impl<T> &Symbol,
                                     StringRef Dynstr) {
  Expected<llvm::StringRef> MaybeName = Symbol.getName(Dynstr);
  if (not MaybeName) {
    auto TheError = MaybeName.takeError();
    revng_log(Log, "Cannot access symbol name: " << TheError);
    consumeError(std::move(TheError));
    return;
  }

  StringRef Name = *MaybeName;
  if (Name.contains('\0')) {
    revng_log(Log,
              "SymbolName contains a NUL character: \"" << Name.str() << "\"");
    return;
  }

  bool IsCode = Symbol.getType() == ELF::STT_FUNC;

  if (shouldIgnoreSymbol(Name))
    return;

  if (Symbol.st_shndx == ELF::SHN_UNDEF) {
    if (IsCode) {
      // Create dynamic function symbol
      Model->ImportedDynamicFunctions[Name.str()];
    } else {
      // TODO: create dynamic global variable
    }
  } else {
    MetaAddress Address = MetaAddress::invalid();

    if (IsCode) {
      Address = relocate(fromPC(Symbol.st_value));
      // TODO: record model::Function::IsDynamic = true
      auto It = Model->Functions.find(Address);
      if (It == Model->Functions.end()) {
        model::Function &Function = Model->Functions[Address];
        Function.Type = model::FunctionType::Regular;
        Function.OriginalName = Name;
      }
    } else {
      Address = relocate(fromGeneric(Symbol.st_value));
      // TODO: create field in segment struct
    }
  }
}

template<typename T>
std::pair<MetaAddress, uint64_t> ELFImporter::ehFrameFromEhFrameHdr() {
  revng_assert(EHFrameHdrAddress);

  auto MaybeEHFrameHdr = File.getFromAddressOn(*EHFrameHdrAddress);
  if (not MaybeEHFrameHdr) {
    revng_log(Log, ".eh_frame_hdr section not available in any segment");
    return { MetaAddress::invalid(), 0 };
  }
  ArrayRef<uint8_t> EHFrameHdr = *MaybeEHFrameHdr;

  using namespace model::Architecture;
  DwarfReader<T> EHFrameHdrReader(toLLVMArchitecture(Architecture),
                                  EHFrameHdr,
                                  *EHFrameHdrAddress);

  uint64_t VersionNumber = EHFrameHdrReader.readNextU8();
  if (VersionNumber != 1) {
    revng_log(Log, "Unexpected version number in .eh_frame: " << VersionNumber);
    return { MetaAddress::invalid(), 0 };
  }

  // ExceptionFrameEncoding
  uint64_t ExceptionFrameEncoding = EHFrameHdrReader.readNextU8();

  // FDEsCountEncoding
  unsigned FDEsCountEncoding = EHFrameHdrReader.readNextU8();

  // LookupTableEncoding
  EHFrameHdrReader.readNextU8();

  Pointer EHFramePointer = EHFrameHdrReader.readPointer(ExceptionFrameEncoding);
  uint64_t FDEsCount = EHFrameHdrReader.readUnsignedValue(FDEsCountEncoding);

  MetaAddress Address = getGenericPointer<T>(EHFramePointer);
  if (Address.isInvalid()) {
    revng_log(Log, "Invalid address of .eh_frame in .eh_frame_hdr");
    return { MetaAddress::invalid(), 0 };
  }

  return { Address, FDEsCount };
}

template<typename T>
void ELFImporter::parseEHFrame(MetaAddress EHFrameAddress,
                               Optional<uint64_t> FDEsCount,
                               Optional<uint64_t> EHFrameSize) {
  if (not FDEsCount and not EHFrameSize) {
    revng_log(Log, "Neither FDE count and .eh_frame size available");
    return;
  }

  // Sometimes the .eh_frame section is present but not mapped in memory. This
  // means it cannot be used at runtime, therefore we can ignore it.
  auto MaybeEHFrame = File.getFromAddressOn(EHFrameAddress);
  if (not MaybeEHFrame)
    return;
  llvm::ArrayRef<uint8_t> EHFrame = *MaybeEHFrame;

  using namespace model::Architecture;
  auto Architecture = toLLVMArchitecture(Model->Architecture);

  DwarfReader<T> EHFrameReader(Architecture, EHFrame, EHFrameAddress);

  // A few fields of the CIE are used when decoding the FDE's.  This struct
  // will cache those fields we need so that we don't have to decode it
  // repeatedly for each FDE that references it.
  struct DecodedCIE {
    Optional<uint32_t> FDEPointerEncoding;
    Optional<uint32_t> LSDAPointerEncoding;
    bool HasAugmentationLength;
  };

  // Map from the start offset of the CIE to the cached data for that CIE.
  DenseMap<uint64_t, DecodedCIE> CachedCIEs;
  unsigned FDEIndex = 0;

  while (!EHFrameReader.eof()
         && ((FDEsCount && FDEIndex < *FDEsCount)
             || (EHFrameSize && EHFrameReader.offset() < *EHFrameSize))) {

    uint64_t StartOffset = EHFrameReader.offset();

    // Read the length of the entry
    uint64_t Length = EHFrameReader.readNextU32();
    if (Length == 0xffffffff)
      Length = EHFrameReader.readNextU64();

    // Compute the end offset of the entry
    uint64_t OffsetAfterLength = EHFrameReader.offset();
    uint64_t EndOffset = OffsetAfterLength + Length;

    // Zero-sized entry, skip it
    if (Length == 0) {
      if (EHFrameReader.offset() != EndOffset) {
        revng_log(Log, ".eh_frame end was expected");
        return;
      }
      continue;
    }

    // Get the entry ID, 0 means it's a CIE, otherwise it's a FDE
    uint32_t ID = EHFrameReader.readNextU32();
    if (ID == 0) {
      // This is a CIE

      // Ensure the version is the one we expect
      uint32_t Version = EHFrameReader.readNextU8();
      if (Version != 1) {
        revng_log(Log, "Unexpected version: " << Version);
        return;
      }

      // Parse a null terminated augmentation string
      SmallString<8> AugmentationString;
      for (uint8_t Char = EHFrameReader.readNextU8(); Char != 0;
           Char = EHFrameReader.readNextU8())
        AugmentationString.push_back(Char);

      // Optionally parse the EH data if the augmentation string says it's
      // there
      if (StringRef(AugmentationString).count("eh") != 0)
        EHFrameReader.readNextU();

      // CodeAlignmentFactor
      EHFrameReader.readULEB128();

      // DataAlignmentFactor
      EHFrameReader.readULEB128();

      // ReturnAddressRegister
      EHFrameReader.readNextU8();

      Optional<uint64_t> AugmentationLength;
      Optional<uint32_t> LSDAPointerEncoding;
      Optional<uint32_t> PersonalityEncoding;
      Optional<uint32_t> FDEPointerEncoding;
      if (!AugmentationString.empty() && AugmentationString.front() == 'z') {
        AugmentationLength = EHFrameReader.readULEB128();

        // Walk the augmentation string to get all the augmentation data.
        for (unsigned I = 1, E = AugmentationString.size(); I != E; ++I) {
          char Char = AugmentationString[I];
          switch (Char) {
          case 'e':
            if (not((I + 1) != E and AugmentationString[I + 1] == 'h')) {
              revng_log(Log, "Expected 'eh' in augmentation string");
              return;
            }
            break;
          case 'L':
            // This is the only information we really care about, all the rest
            // is processed just so we can get here
            if (not LSDAPointerEncoding)
              LSDAPointerEncoding = EHFrameReader.readNextU8();
            else
              revng_log(Log, "Duplicate LSDA encoding. Ignoroing.");

            break;
          case 'P': {
            if (PersonalityEncoding) {
              revng_log(Log, "Duplicate personality. Ignoring.");
              break;
            }
            PersonalityEncoding = EHFrameReader.readNextU8();
            // Personality
            Pointer Personality;
            Personality = EHFrameReader.readPointer(*PersonalityEncoding);
            auto PersonalityPtr = getCodePointer<T>(Personality);
            logAddress(Log, "Personality function: ", PersonalityPtr);

            // Register in the model for exploration
            Model->ExtraCodeAddresses.insert(PersonalityPtr);
            break;
          }
          case 'R':
            if (FDEPointerEncoding) {
              revng_log(Log, "Duplicate FDE encoding. Ignoring.");
              break;
            }
            FDEPointerEncoding = EHFrameReader.readNextU8();
            break;
          case 'z':
            revng_log(Log, "'z' must be first in the augmentation string");
            return;
          }
        }
      }

      // Cache this entry
      CachedCIEs[StartOffset] = { FDEPointerEncoding,
                                  LSDAPointerEncoding,
                                  AugmentationLength.hasValue() };

    } else {
      // This is an FDE
      FDEIndex++;

      // The CIE pointer for an FDE is the same location as the ID which we
      // already read
      uint64_t CIEOffset = OffsetAfterLength - ID;

      // Ensure we already met this CIE
      auto CIEIt = CachedCIEs.find(CIEOffset);
      if (CIEIt == CachedCIEs.end()) {
        revng_log(Log, "Couldn't find CIE at offset in to __eh_frame section");
        return;
      }

      // Ensure we have at least the pointer encoding
      const DecodedCIE &CIE = CIEIt->getSecond();
      if (not CIE.FDEPointerEncoding) {
        revng_log(Log, "FDE references CIE which did not set pointer encoding");
        return;
      }

      // PCBegin
      auto PCBeginPointer = EHFrameReader.readPointer(*CIE.FDEPointerEncoding);
      MetaAddress PCBegin = getGenericPointer<T>(PCBeginPointer);

      // PCRange
      EHFrameReader.readPointer(*CIE.FDEPointerEncoding);

      if (CIE.HasAugmentationLength)
        EHFrameReader.readULEB128();

      // Decode the LSDA if the CIE augmentation string said we should.
      if (CIE.LSDAPointerEncoding) {
        auto LSDAPointer = EHFrameReader.readPointer(*CIE.LSDAPointerEncoding);
        parseLSDA<T>(PCBegin, getGenericPointer<T>(LSDAPointer));
      }
    }

    // Skip all the remaining parts
    EHFrameReader.moveTo(EndOffset);
  }
}

template<typename T>
void ELFImporter::parseLSDA(MetaAddress FDEStart, MetaAddress LSDAAddress) {
  logAddress(Log, "LSDAAddress: ", LSDAAddress);

  auto MaybeLSDA = File.getFromAddressOn(LSDAAddress);
  if (not MaybeLSDA) {
    revng_log(Log, "LSDA not available in any segment");
    return;
  }
  llvm::ArrayRef<uint8_t> LSDA = *MaybeLSDA;

  using namespace model::Architecture;
  auto Architecture = toLLVMArchitecture(Model->Architecture);
  DwarfReader<T> LSDAReader(Architecture, LSDA, LSDAAddress);

  uint32_t LandingPadBaseEncoding = LSDAReader.readNextU8();
  MetaAddress LandingPadBase = MetaAddress::invalid();
  if (LandingPadBaseEncoding != dwarf::DW_EH_PE_omit) {
    auto LandingPadBasePointer = LSDAReader.readPointer(LandingPadBaseEncoding);
    LandingPadBase = getGenericPointer<T>(LandingPadBasePointer);
  } else {
    LandingPadBase = FDEStart;
  }

  logAddress(Log, "LandingPadBase: ", LandingPadBase);

  uint32_t TypeTableEncoding = LSDAReader.readNextU8();
  if (TypeTableEncoding != dwarf::DW_EH_PE_omit)
    LSDAReader.readULEB128();

  uint32_t CallSiteTableEncoding = LSDAReader.readNextU8();
  uint64_t CallSiteTableLength = LSDAReader.readULEB128();
  uint64_t CallSiteTableEnd = LSDAReader.offset() + CallSiteTableLength;

  while (LSDAReader.offset() < CallSiteTableEnd) {
    // InstructionStart
    LSDAReader.readPointer(CallSiteTableEncoding);

    // InstructionEnd
    LSDAReader.readPointer(CallSiteTableEncoding);

    // LandingPad
    Pointer LandingPadPointer = LSDAReader.readPointer(CallSiteTableEncoding,
                                                       LandingPadBase);
    MetaAddress LandingPad = getCodePointer<T>(LandingPadPointer);

    // Action
    LSDAReader.readULEB128();

    if (LandingPad.isValid()) {
      auto &ExtraCodeAddresses = Model->ExtraCodeAddresses;
      if (ExtraCodeAddresses.count(LandingPad) == 0)
        logAddress(Log, "New landing pad found: ", LandingPad);

      ExtraCodeAddresses.insert(LandingPad);
    }
  }
}

template<typename T, bool HasAddend>
struct RelocationHelper {
  static uint64_t getAddend(llvm::object::Elf_Rel_Impl<T, HasAddend>);
};

template<typename T>
struct RelocationHelper<T, true> {
  static uint64_t getAddend(llvm::object::Elf_Rel_Impl<T, true> Relocation) {
    return Relocation.r_addend;
  }
};

template<typename T>
struct RelocationHelper<T, false> {
  static uint64_t getAddend(llvm::object::Elf_Rel_Impl<T, false>) { return 0; }
};

template<typename T, bool HasAddend>
void ELFImporter::registerRelocations(Elf_Rel_Array<T, HasAddend> Relocations,
                                      const FilePortion2 &Dynsym,
                                      const FilePortion2 &Dynstr) {
  using namespace llvm::object;
  using Elf_Rel = Elf_Rel_Impl<T, HasAddend>;
  using Elf_Sym = Elf_Sym_Impl<T>;

  model::Segment *LowestSegment = nullptr;
  if (auto It = Model->Segments.begin(); It != Model->Segments.end())
    LowestSegment = &*It;

  ArrayRef<Elf_Sym> Symbols;
  if (Dynsym.isAvailable())
    Symbols = Dynsym.extractAs<Elf_Sym>();

  for (Elf_Rel Relocation : Relocations) {
    auto Type = static_cast<unsigned char>(Relocation.getType(false));
    uint64_t Addend = RelocationHelper<T, HasAddend>::getAddend(Relocation);
    MetaAddress Address = relocate(fromGeneric(Relocation.r_offset));

    StringRef SymbolName;
    uint64_t SymbolSize = 0;
    unsigned char SymbolType = llvm::ELF::STT_NOTYPE;
    if (Dynsym.isAvailable() and Dynstr.isAvailable()) {
      uint32_t SymbolIndex = Relocation.getSymbol(false);
      if (not(SymbolIndex < Symbols.size())) {
        revng_log(Log,
                  "Invalid symbol index "
                    << SymbolIndex << ". "
                    << "Symbol count: " << Symbols.size());
      }
      const Elf_Sym &Symbol = Symbols[SymbolIndex];
      auto MaybeName = Symbol.getName(Dynstr.extractString());
      if (MaybeName)
        SymbolName = *MaybeName;
      SymbolSize = Symbol.st_size;
      SymbolType = Symbol.getType();
    }

    using namespace model::RelocationType;
    auto RelocationType = fromELFRelocation(Model->Architecture, Type);

    auto RelocationName = getELFRelocationTypeName(TheBinary.getEMachine(),
                                                   Type);
    if (RelocationType == Invalid) {
      revng_log(Log, "Ignoring unknown relocation: " << RelocationName);
      continue;
    }

    model::Relocation NewRelocation(Address, RelocationType, Addend);

    bool HasName = SymbolName.size() != 0;
    bool IsBaseRelative = isELFRelocationBaseRelative(Architecture, Type);

    if (HasName and IsBaseRelative) {
      revng_log(Log,
                "We found a base-relative relocation ("
                  << RelocationName << ") associated to a symbol, ignoring.");
    } else if (not HasName and not IsBaseRelative) {
      if (Log.isEnabled()) {
        Log << "We found a non-base-relative relocation (" << RelocationName
            << ") not associated to a symbol, ignoring." << DoLog;
      }
    } else if (HasName) {
      // Symbol-relative relcation
      if (SymbolType == ELF::STT_FUNC) {
        auto It = Model->ImportedDynamicFunctions.find(SymbolName.str());
        if (It != Model->ImportedDynamicFunctions.end()) {
          auto &Relocations = It->Relocations;
          NewRelocation.verify(true);
          Relocations.insert(NewRelocation);
        }
      } else {
        // TODO: register relocation for dynamic global variable
      }
    } else {
      // Base-relative relocation
      if (LowestSegment != nullptr) {
        NewRelocation.verify(true);
        LowestSegment->Relocations.insert(NewRelocation);
      } else {
        revng_log(Log,
                  "Found a base-relative relocation, but no segment is "
                  "available! Ignoring.");
      }
    }
  }
}

Error importELF(TupleTree<model::Binary> &Model,
                const object::ELFObjectFileBase &TheBinary,
                uint64_t PreferredBaseAddress) {
  ELFImporter Importer(Model, TheBinary, PreferredBaseAddress);

  using namespace model::Architecture;
  bool IsLittleEndian = isLittleEndian(Model->Architecture);
  size_t PointerSize = getPointerSize(Model->Architecture);
  bool HasRelocationAddend = hasELFRelocationAddend(Model->Architecture);

  if (PointerSize == 4) {
    if (IsLittleEndian) {
      if (HasRelocationAddend) {
        return Importer.import<object::ELF32LE, true>();
      } else {
        return Importer.import<object::ELF32LE, false>();
      }
    } else {
      if (HasRelocationAddend) {
        return Importer.import<object::ELF32BE, true>();
      } else {
        return Importer.import<object::ELF32BE, false>();
      }
    }
  } else if (PointerSize == 8) {
    if (IsLittleEndian) {
      if (HasRelocationAddend) {
        return Importer.import<object::ELF64LE, true>();
      } else {
        return Importer.import<object::ELF64LE, false>();
      }
    } else {
      if (HasRelocationAddend) {
        return Importer.import<object::ELF64BE, true>();
      } else {
        return Importer.import<object::ELF64BE, false>();
      }
    }
  } else {
    revng_abort("Unexpect address size");
  }
}
