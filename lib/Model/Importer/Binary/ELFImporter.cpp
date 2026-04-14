//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <cstdint>
#include <optional>

#include "llvm/ADT/DenseMap.h"
#include "llvm/Object/ELF.h"
#include "llvm/Object/ELFObjectFile.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/Progress.h"

#include "revng/ABI/DefaultFunctionPrototype.h"
#include "revng/Model/Binary.h"
#include "revng/Model/IRHelpers.h"
#include "revng/Model/Importer/Binary/BinaryDescriptor.h"
#include "revng/Model/Importer/Binary/BinaryImporter.h"
#include "revng/Model/Importer/Binary/BinaryImporterHelper.h"
#include "revng/Model/Importer/Binary/Options.h"
#include "revng/Model/Importer/DebugInfo/DwarfImporter.h"
#include "revng/Model/Pass/AllPasses.h"
#include "revng/Model/RawBinaryView.h"
#include "revng/Support/CommandLine.h"
#include "revng/Support/Configuration.h"
#include "revng/Support/Debug.h"
#include "revng/Support/Error.h"
#include "revng/Support/LDDTree.h"
#include "revng/TupleTree/TupleTree.h"

#include "DwarfReader.h"
#include "ELFImporter.h"
#include "FindMissingTypes.h"
#include "Importers.h"
#include "MIPSELFImporter.h"

using namespace llvm;
using namespace llvm::object;

Logger ELFImporterLog("elf-importer");

template<typename A, typename B>
static bool hasFlag(A Flag, B Value) {
  return (Flag & Value) != 0;
}

FilePortion::FilePortion(const RawBinaryView &File) :
  File(File),
  HasAddress(false),
  HasSize(false),
  Size(0),
  Address(MetaAddress::invalid()) {
}

void FilePortion::setAddress(MetaAddress Address) {
  HasAddress = true;
  this->Address = Address;
}

void FilePortion::setSize(uint64_t Size) {
  HasSize = true;
  this->Size = Size;
}

MetaAddress FilePortion::addressAtOffset(uint64_t Offset) {
  if (not(HasAddress and HasSize and Offset <= Size))
    return MetaAddress::invalid();
  return Address + Offset;
}

template<typename T>
MetaAddress FilePortion::addressAtIndex(uint64_t Index) {
  uint64_t Offset = Index * sizeof(T);
  if (not(HasAddress and HasSize and Offset <= Size))
    return MetaAddress::invalid();
  return Address + Offset;
}

bool FilePortion::isAvailable() const {
  return HasAddress;
}

bool FilePortion::isExact() const {
  revng_assert(HasAddress);
  return HasSize;
}

StringRef FilePortion::extractString() const {
  auto Data = extractData();

  const char *AsChar = reinterpret_cast<const char *>(Data.data());
  return StringRef(AsChar, Data.size());
}

template<typename T>
ArrayRef<T> FilePortion::extractAs() const {
  auto Data = extractData();

  const size_t TypeSize = sizeof(T);
  if (Data.size() % TypeSize != 0) {
    revng_log(ELFImporterLog,
              "Warning: cannot extract objects of size "
                << TypeSize
                << ". File portion size is not a multiple: " << Data.size());
    return {};
  }

  return ArrayRef<T>(reinterpret_cast<const T *>(Data.data()),
                     Data.size() / TypeSize);
}

ArrayRef<uint8_t> FilePortion::extractData() const {
  revng_assert(HasAddress);

  if (HasSize) {
    auto MaybeData = File.getByAddress(Address, Size);
    if (MaybeData) {
      return *MaybeData;
    } else {
      revng_log(ELFImporterLog,
                "Cannot access address " << Address.toString() << " and size "
                                         << Size);
      return {};
    }
  } else {
    auto MaybeData = File.getFromAddressOn(Address);
    if (MaybeData) {
      return *MaybeData;
    } else {
      revng_log(ELFImporterLog, "Cannot access address " << Address.toString());
      return {};
    }
  }
}

static bool shouldIgnoreSymbol(StringRef Name) {
  return Name == "$a" or Name == "$d";
}

static bool endsWith(StringRef String, char Last) {
  return not String.empty() and String.back() == Last;
}

static llvm::StringRef extractNullTerminatedStringAt(llvm::StringRef Source,
                                                     uint64_t Offset) {
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

template<typename T, bool HasAddend>
uint64_t symbolsCount(const FilePortion &Relocations) {
  using Elf_Rel = llvm::object::Elf_Rel_Impl<T, HasAddend>;

  if (not Relocations.isAvailable() or not Relocations.isExact())
    return 0;

  uint32_t SymbolsCount = 0;

  for (Elf_Rel Relocation : Relocations.extractAs<Elf_Rel>()) {
    SymbolsCount = std::max(SymbolsCount, Relocation.getSymbol(false) + 1);
  }

  return SymbolsCount;
}

template<typename T, bool HasAddend>
Error ELFImporter<T, HasAddend>::import(const ImporterOptions &Options) {
  revng_log(ELFImporterLog, "Starting ELF import");
  LoggerIndent Indent(ELFImporterLog);

  llvm::Task Task(14, "Import ELF");
  Task.advance("Parse ELF", true);

  // Parse the ELF file
  auto TheELFOrErr = object::ELFFile<T>::create(TheBinary.ObjectFile.getData());
  if (not TheELFOrErr)
    return TheELFOrErr.takeError();
  object::ELFFile<T> &TheELF = *TheELFOrErr;

  // Set operating system from the ELF OSABI field
  {
    unsigned char OSABI = TheELF.getHeader().e_ident[llvm::ELF::EI_OSABI];
    if (OSABI == llvm::ELF::ELFOSABI_NONE or OSABI == llvm::ELF::ELFOSABI_LINUX)
      Model->OperatingSystem() = model::OperatingSystem::Linux;
  }

  // Parse segments
  Task.advance("Parse segments", true);
  parseSegments(TheELF);

  // Set default ABI
  if (Model->DefaultABI() == model::ABI::Invalid) {
    revng_assert(Model->Architecture() != model::Architecture::Invalid);
    if (auto ABI = model::ABI::getDefaultForELF(Model->Architecture())) {
      Model->DefaultABI() = ABI.value();
    } else {
      auto ArchName = model::Architecture::getName(Model->Architecture()).str();
      return revng::createError("Unsupported architecture for ELF: "
                                + ArchName);
    }
  }

  // BaseAddress makes sense only for shared (relocatable, PIC) objects
  auto Type = TheELF.getHeader().e_type;
  ImporterOptions AdjustedOptions = ImporterOptions{
    .BaseAddress = Options.BaseAddress,
    .DebugInfo = Options.DebugInfo,
    .EnableRemoteDebugInfo = Options.EnableRemoteDebugInfo,
    .AdditionalDebugInfoPaths = Options.AdditionalDebugInfoPaths
  };

  if (not(Type == ELF::ET_DYN or Type == ELF::ET_EXEC))
    return revng::createError("Only ELF executables and ELF dynamic libraries "
                              "are supported");

  // Look for static or dynamic symbols and relocations
  ConstElf_Shdr *SymtabShdr = nullptr;
  std::optional<MetaAddress> EHFrameAddress;
  std::optional<uint64_t> EHFrameSize;

  Task.advance("Parse sections", true);

  SmallVector<Section, 16> Sections;
  auto ELFSections = TheELF.sections();
  if (auto Error = ELFSections.takeError()) {
    // TODO: emit a diagnostic message for the user.
    revng_log(ELFImporterLog, "Sections unavailable: " << Error);
    llvm::consumeError(std::move(Error));
  } else {
    for (ConstElf_Shdr &SectionHeader : *ELFSections) {
      // Obtain the section name
      StringRef SectionName;

      auto MaybeSectionName = TheELF.getSectionName(SectionHeader);
      if (auto Error = MaybeSectionName.takeError()) {
        revng_log(ELFImporterLog, "Cannot access section name: " << Error);
        consumeError(std::move(Error));
      } else {
        SectionName = *MaybeSectionName;
      }

      // Collect section names
      if (hasFlag(SectionHeader.sh_flags, ELF::SHF_ALLOC)) {
        bool CanContainCode = hasFlag(SectionHeader.sh_flags,
                                      ELF::SHF_EXECINSTR);
        auto SectionStart = relocate(fromGeneric(SectionHeader.sh_addr));
        uint64_t Size = SectionHeader.sh_size;
        auto SectionEnd = SectionStart + Size;

        // Note: we will discard overlapping sections later on, in
        // populateSegmentTypeStruct
        if (SectionStart.isValid() and SectionEnd.isValid()
            and SectionStart.addressLowerThan(SectionEnd)) {
          Section NewSection(SectionStart, SectionHeader.sh_size);

          NewSection.Name = SectionName.str();
          NewSection.CanContainCode = CanContainCode;
          Sections.push_back(std::move(NewSection));
        }
      }

      // Handle well-known sections
      if (SectionName == ".symtab") {
        // TODO: check dedicated field in section header
        if (SymtabShdr == nullptr)
          SymtabShdr = &SectionHeader;
        else
          revng_log(ELFImporterLog, "Multiple .symtab. Ignoring.");
      } else if (SectionName == ".eh_frame") {
        if (not EHFrameAddress) {
          EHFrameAddress = relocate(fromGeneric(SectionHeader.sh_addr));
          EHFrameSize = static_cast<uint64_t>(SectionHeader.sh_size);
        } else {
          revng_log(ELFImporterLog, "Duplicate .eh_frame. Ignoring.");
        }
      } else if (SectionName == ".dynamic") {
        if (not DynamicAddress)
          DynamicAddress = relocate(fromGeneric(SectionHeader.sh_addr));
        else
          revng_log(ELFImporterLog, "Duplicate .dynamic. Ignoring.");
      }
    }
  }

  Task.advance("Parse static symbols", true);
  parseSymbols(TheELF, SymtabShdr);

  const auto &ElfHeader = TheELF.getHeader();
  if (ElfHeader.e_entry != 0)
    setEntryPoint(relocate(fromPC(ElfHeader.e_entry)));

  // Parse program headers
  Task.advance("Parse program headers", true);
  parseProgramHeaders(TheELF);

  std::optional<uint64_t> FDEsCount;
  if (EHFrameHdrAddress) {
    MetaAddress Address = MetaAddress::invalid();

    std::tie(Address, FDEsCount) = ehFrameFromEhFrameHdr();
    if (Address.isValid()) {
      if (EHFrameAddress and *EHFrameAddress != Address) {
        revng_log(ELFImporterLog,
                  "Incoherent .eh_frame information: .eh_frame is at "
                    << EHFrameAddress->toString()
                    << " while .eh_frame_hdr reports " << Address.toString());
      }

      EHFrameAddress = Address;
    }
  }

  Task.advance("Parse .eh_frame", true);
  if (EHFrameAddress and EHFrameAddress->isValid())
    parseEHFrame(*EHFrameAddress, FDEsCount, EHFrameSize);

  // Parse the .dynamic table
  Task.advance("Parse .dynamic", true);
  auto DynamicEntries = TheELF.dynamicEntries();
  if (auto Error = DynamicEntries.takeError()) {
    revng_log(ELFImporterLog, "Cannot access dynamic entries: " << Error);
    consumeError(std::move(Error));
  } else {
    revng_log(ELFImporterLog, "Parsing .dynamic");
    LoggerIndent Indent(ELFImporterLog);

    SmallVector<uint64_t, 10> NeededLibraryNameOffsets;

    // TODO: use std::optional
    DynstrPortion = std::make_unique<FilePortion>(File);
    DynsymPortion = std::make_unique<FilePortion>(File);
    ReldynPortion = std::make_unique<FilePortion>(File);
    RelpltPortion = std::make_unique<FilePortion>(File);
    GotPortion = std::make_unique<FilePortion>(File);
    bool IsX86 = Model->Architecture() == model::Architecture::x86;
    bool IsMIPS = (Model->Architecture() == model::Architecture::mips
                   or Model->Architecture() == model::Architecture::mipsel);

    using Elf_Dyn = const typename object::ELFFile<T>::Elf_Dyn;
    for (Elf_Dyn &DynamicTag : *DynamicEntries) {
      parseDynamicTag(DynamicTag.getTag(),
                      DynamicTag.getVal(),
                      DynamicTag.getPtr(),
                      NeededLibraryNameOffsets);
    }

    StringRef Dynstr;

    if (DynstrPortion->isAvailable()) {
      Dynstr = DynstrPortion->extractString();
      auto Inserter = Model->ImportedLibraries().batch_insert();
      for (auto Offset : NeededLibraryNameOffsets) {
        StringRef LibraryName = extractNullTerminatedStringAt(Dynstr, Offset);
        revng_assert(not endsWith(LibraryName, '\0'));
        Inserter.insert(LibraryName.data());
      }
    }

    // Collect symbols count and code pointers in image base-relative
    // relocations

    if (not SymbolsCount) {
      SymbolsCount = std::max(symbolsCount<T, HasAddend>(*ReldynPortion.get()),
                              symbolsCount<T, HasAddend>(*RelpltPortion.get()));
    }

    revng_log(ELFImporterLog, "SymbolsCount: " << SymbolsCount.value_or(0));
    revng_log(ELFImporterLog,
              "DynsymPortion->isAvailable(): " << DynsymPortion->isAvailable());

    // Collect function addresses contained in dynamic symbols
    if (SymbolsCount and *SymbolsCount > 0 and DynsymPortion->isAvailable()) {
      Task.advance("Parse dynamic symbols", true);

      using Elf_Sym = llvm::object::Elf_Sym_Impl<T>;
      DynsymPortion->setSize(*SymbolsCount * sizeof(Elf_Sym));

      ArrayRef<Elf_Sym> Symbols = DynsymPortion->extractAs<Elf_Sym>();

      for (Elf_Sym Symbol : Symbols)
        parseDynamicSymbol(Symbol, Dynstr);

      using Elf_Rel = llvm::object::Elf_Rel_Impl<T, HasAddend>;
      if (ReldynPortion->isAvailable()) {
        registerRelocations(".rela.dyn",
                            ReldynPortion->extractAs<Elf_Rel>(),
                            *DynsymPortion.get(),
                            *DynstrPortion.get());
      }

      auto SetCanonicalValue = [this](model::Register::Values Register,
                                      uint64_t Value) {
        for (model::Segment &Segment : Model->Segments())
          if (Segment.IsExecutable())
            Segment.CanonicalRegisterValues()[Register].Value() = Value;
      };

      if (GotPortion->isAvailable()) {
        if (IsX86) {
          SetCanonicalValue(model::Register::ebx_x86,
                            GotPortion->address().address());
        } else if (IsMIPS) {
          SetCanonicalValue(model::Register::gp_mips,
                            GotPortion->address().address() + 0x7ff0);
        }
      }

      if (RelpltPortion->isAvailable()) {
        registerRelocations(".rela.plt",
                            RelpltPortion->extractAs<Elf_Rel>(),
                            *DynsymPortion.get(),
                            *DynstrPortion.get());
      }
    }
  }

  // Dynamic symbols harvested too, segment type creation can be finalized.
  // Do not replace it, if `Type` is present (may have been added by the user).
  Task.advance("Parse segment struct from data symbols", true);
  for (auto &Segment : Model->Segments()) {
    if (Segment.Type().isEmpty()) {
      Segment.Type() = populateSegmentTypeStruct(*Model,
                                                 Segment,
                                                 DataSymbols,
                                                 Sections,
                                                 Segment.IsExecutable());
    }
  }

  // Create a default prototype

  auto &Ptr = *Model.get();
  Model->DefaultPrototype() = abi::registerDefaultFunctionPrototype(Ptr);

  if (AdjustedOptions.DebugInfo != DebugInfoLevel::No) {

    auto Handler = [&](const auto &ObjectFile) {
      Task.advance("Identifying dependencies", true);
      std::optional<LDDTree>
        MaybeDependencies = identifyDependencies(ObjectFile,
                                                 TheBinary.canonicalPath());

      bool HasRoot = MaybeDependencies.has_value();
      const revng::RootEntry *Root = HasRoot ? MaybeDependencies->Root :
                                               nullptr;

      // Import Dwarf
      Task.advance("Parse debug info", true);
      {
        auto ImportLogger = importLogger(TheBinary.canonicalPath());
        DwarfImporter Importer(Model, std::nullopt);
        Importer.import(Root, TheBinary, AdjustedOptions);
      }

      // Now we try to find missing types in the dependencies
      Task.advance("Find missing types from debug info", true);

      if (HasRoot) {
        findMissingTypes<ELFBinary, DwarfImporter>(*MaybeDependencies,
                                                   Options,
                                                   Logger,
                                                   Model);
      }
    };

    auto *GenericObjectFile = &TheBinary.ObjectFile;
    if (auto *ObjectFile = dyn_cast<ELF32LEObjectFile>(GenericObjectFile))
      Handler(*ObjectFile);
    else if (auto *ObjectFile = dyn_cast<ELF32BEObjectFile>(GenericObjectFile))
      Handler(*ObjectFile);
    else if (auto *ObjectFile = dyn_cast<ELF64LEObjectFile>(GenericObjectFile))
      Handler(*ObjectFile);
    else if (auto *ObjectFile = dyn_cast<ELF64BEObjectFile>(GenericObjectFile))
      Handler(*ObjectFile);
    else
      revng_abort();
  }

  Task.advance("Flatten primitive typedefs", true);
  model::flattenPrimitiveTypedefs(Model);

  Task.advance("Deduplicate colliding names", true);
  model::deduplicateCollidingNames(Model);

  return Error::success();
}

using Libs = SmallVectorImpl<uint64_t>;
template<typename T, bool HasAddend>
void ELFImporter<T, HasAddend>::parseDynamicTag(uint64_t Tag,
                                                uint64_t Val,
                                                uint64_t Pointer,
                                                Libs &LibrariesOffsets) {
  MetaAddress GenericAddress = relocate(fromGeneric(Pointer));
  MetaAddress PCAddress = relocate(fromPC(Pointer));
  switch (Tag) {
  case ELF::DT_NEEDED:
    LibrariesOffsets.push_back(Val);
    break;

  case ELF::DT_STRTAB:
    DynstrPortion->setAddress(GenericAddress);
    break;

  case ELF::DT_STRSZ:
    DynstrPortion->setSize(Val);
    break;

  case ELF::DT_SYMTAB:
    DynsymPortion->setAddress(GenericAddress);
    break;

  case ELF::DT_JMPREL:
    RelpltPortion->setAddress(GenericAddress);
    break;

  case ELF::DT_PLTRELSZ:
    RelpltPortion->setSize(Val);
    break;

  case ELF::DT_REL:
  case ELF::DT_RELA:
    if (Tag != (HasAddend ? ELF::DT_RELA : ELF::DT_REL)) {
      if (Tag == ELF::DT_RELA)
        revng_log(ELFImporterLog, "Unexpected addend in relocation");
      else
        revng_log(ELFImporterLog, "Addend was expected in relocation");
    }
    ReldynPortion->setAddress(GenericAddress);
    break;

  case ELF::DT_RELSZ:
  case ELF::DT_RELASZ:
    if (Tag != (HasAddend ? ELF::DT_RELASZ : ELF::DT_RELSZ)) {
      if (Tag == ELF::DT_RELASZ)
        revng_log(ELFImporterLog, "Unexpected addend in relocation");
      else
        revng_log(ELFImporterLog, "Addend was expected in relocation");
    }
    ReldynPortion->setSize(Val);
    break;

  case ELF::DT_PLTGOT:
    GotPortion->setAddress(GenericAddress);
    break;

  case ELF::DT_INIT:
  case ELF::DT_FINI:
    revng_assert(PCAddress.isValid());
    registerFunctionEntry(PCAddress);
    break;

  default:
    parseTargetDynamicTags(Tag, GenericAddress, LibrariesOffsets, Val);
    break;
  }
}

// TODO: we might want to return an error from here so we can propagate it
//       further up.
template<typename T, bool HasAddend>
void ELFImporter<T, HasAddend>::parseSymbols(object::ELFFile<T> &TheELF,
                                             ConstElf_Shdr *SymtabShdr) {
  // Check if we found a symbol table
  if (SymtabShdr == nullptr or SymtabShdr->sh_link == 0)
    return;

  // Obtain a reference to the string table
  auto Strtab = TheELF.getSection(SymtabShdr->sh_link);
  if (auto Error = Strtab.takeError()) {
    revng_log(ELFImporterLog, "Cannot find .strtab: " << Error);
    consumeError(std::move(Error));
    return;
  }

  auto StrtabArray = TheELF.getSectionContents(**Strtab);
  if (auto Error = StrtabArray.takeError()) {
    revng_log(ELFImporterLog, "Cannot access .strtab: " << Error);
    consumeError(std::move(Error));
    return;
  }

  StringRef StrtabContent(reinterpret_cast<const char *>(StrtabArray->data()),
                          StrtabArray->size());

  // Collect symbol names
  auto ELFSymbols = TheELF.symbols(SymtabShdr);
  if (auto Error = ELFSymbols.takeError()) {
    revng_log(ELFImporterLog, "Cannot get symbols: " << Error);
    consumeError(std::move(Error));
    return;
  }

  for (auto &Symbol : *ELFSymbols) {
    auto MaybeName = expectedToOptional(Symbol.getName(StrtabContent));

    if ((MaybeName and shouldIgnoreSymbol(*MaybeName))
        or (Symbol.st_shndx == ELF::SHN_UNDEF))
      continue;

    MetaAddress Address = MetaAddress::invalid();
    bool IsCode = Symbol.getType() == ELF::STT_FUNC;
    bool IsDataObject = Symbol.getType() == ELF::STT_OBJECT;
    uint64_t Size = Symbol.st_size;

    if (IsCode)
      Address = relocate(fromPC(Symbol.st_value));
    else
      Address = relocate(fromGeneric(Symbol.st_value));

    if (IsCode) {
      revng_assert(Address.isValid());
      if (Model->Functions().tryGet(Address) == nullptr) {
        auto *Function = registerFunctionEntry(Address);
        if (Function != nullptr and MaybeName and MaybeName->size() > 0) {
          Function->Name() = *MaybeName;
        }
      }
    } else if (IsDataObject and Size > 0) {
      auto IsSameAddress = [Address](const auto &E) {
        return Address == E.Address;
      };
      if (llvm::count_if(DataSymbols, IsSameAddress) == 0)
        DataSymbols.emplace_back(Address, Size, *MaybeName);
    }
  }
}

template<typename T, bool HasAddend>
void ELFImporter<T, HasAddend>::parseSegments(ELFFile<T> &TheELF) {
  auto ProgHeaders = TheELF.program_headers();
  if (auto Error = ProgHeaders.takeError()) {
    revng_log(ELFImporterLog, "Cannot access program headers: " << Error);
    consumeError(std::move(Error));
    return;
  }

  for (auto &ProgramHeader : *ProgHeaders) {
    if (ProgramHeader.p_type == ELF::PT_LOAD) {
      auto Start = relocate(fromGeneric(ProgramHeader.p_vaddr));
      auto EndVirtualAddress = Start + u64(ProgramHeader.p_memsz);
      if (Start.isInvalid() or EndVirtualAddress.isInvalid()) {
        revng_log(ELFImporterLog, "Invalid segment found");
        continue;
      }

      auto VirtualSize = ProgramHeader.p_memsz;
      if (VirtualSize == 0) {
        revng_log(ELFImporterLog, "Ignoring zero-sized segment");
        continue;
      }

      if (VirtualSize >= std::numeric_limits<int64_t>::max()) {
        // TODO: emit a diagnostic message for the user.
        revng_log(ELFImporterLog,
                  "Ignoring too large segment: " << VirtualSize << " bytes");
        continue;
      }

      model::Segment NewSegment({ Start, ProgramHeader.p_memsz });

      // TODO: do the following unconditionally once the old pipeline has been
      //       dropped.
      if (TheBinary.Reference.isValid())
        NewSegment.Binary() = TheBinary.Reference;
      NewSegment.StartOffset() = ProgramHeader.p_offset;

      auto MaybeEndOffset = (OverflowSafeInt(u64(ProgramHeader.p_offset))
                             + u64(ProgramHeader.p_filesz));
      if (not MaybeEndOffset) {
        // TODO: emit a diagnostic message for the user.
        revng_log(ELFImporterLog,
                  "Invalid segment found: overflow in computing end offset");
        continue;
      }
      NewSegment.FileSize() = ProgramHeader.p_filesz;

      NewSegment.IsReadable() = hasFlag(ProgramHeader.p_flags, ELF::PF_R);
      NewSegment.IsWriteable() = hasFlag(ProgramHeader.p_flags, ELF::PF_W);
      NewSegment.IsExecutable() = hasFlag(ProgramHeader.p_flags, ELF::PF_X);

      NewSegment.verify(true);

      Model->Segments().insert(std::move(NewSegment));
    }
  }

  processSegments();
}

template<typename T, bool HasAddend>
void ELFImporter<T, HasAddend>::parseProgramHeaders(ELFFile<T> &TheELF) {
  using Elf_Phdr = const typename object::ELFFile<T>::Elf_Phdr;

  Elf_Phdr *DynamicPhdr = nullptr;

  auto ProgHeaders = TheELF.program_headers();
  if (auto Error = ProgHeaders.takeError()) {
    revng_log(ELFImporterLog, "Cannot access program headers: " << Error);
    consumeError(std::move(Error));
    return;
  }

  for (Elf_Phdr &ProgramHeader : *ProgHeaders) {
    switch (ProgramHeader.p_type) {
    case ELF::PT_LOAD: {
      // Already processed
    } break;

    case ELF::PT_GNU_EH_FRAME:
      if (not EHFrameHdrAddress)
        EHFrameHdrAddress = relocate(fromGeneric(ProgramHeader.p_vaddr));
      else
        revng_log(ELFImporterLog, "Multiple PT_GNU_EH_FRAME. Ignoring.");
      break;

    case ELF::PT_DYNAMIC:
      if (DynamicPhdr != nullptr) {
        revng_log(ELFImporterLog, "Duplicate .dynamic program header");
        break;
      }
      DynamicPhdr = &ProgramHeader;
      MetaAddress DynamicPhdrMA = relocate(fromGeneric(DynamicPhdr->p_vaddr));

      if (DynamicAddress and DynamicPhdrMA != *DynamicAddress) {
        revng_log(ELFImporterLog,
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

  if ((DynamicPhdr != nullptr) != (DynamicAddress.has_value())) {
    revng_log(ELFImporterLog, "Invalid .dynamic/PT_DYNAMIC");
    DynamicPhdr = nullptr;
    DynamicAddress = {};
  }
}

template<typename T, bool HasAddend>
void ELFImporter<T, HasAddend>::parseDynamicSymbol(Elf_Sym_Impl<T> &Symbol,
                                                   StringRef Dynstr) {
  Expected<llvm::StringRef> MaybeName = Symbol.getName(Dynstr);
  if (auto TheError = MaybeName.takeError()) {
    revng_log(ELFImporterLog, "Cannot access symbol name: " << TheError);
    consumeError(std::move(TheError));
    return;
  }

  StringRef Name = *MaybeName;

  revng_log(ELFImporterLog, "Considering dynamic symbol " << Name);
  LoggerIndent Indent(ELFImporterLog);

  if (Name.contains('\0')) {
    revng_log(ELFImporterLog,
              "SymbolName contains a NUL character: \"" << Name.str() << "\"");
    return;
  }

  bool IsCode = Symbol.getType() == ELF::STT_FUNC;
  bool IsDataObject = Symbol.getType() == ELF::STT_OBJECT;

  if (shouldIgnoreSymbol(Name)) {
    revng_log(ELFImporterLog, "Ignoring");
    return;
  }

  if (Symbol.st_shndx == ELF::SHN_UNDEF) {
    revng_log(ELFImporterLog, "This is an imported symbol");
    if (IsCode) {
      // Create dynamic function symbol
      Model->ImportedDynamicFunctions()[Name.str()];
    } else {
      // TODO: create dynamic global variable
      revng_log(ELFImporterLog, "Ignoring non-code");
    }
  } else {
    revng_log(ELFImporterLog, "It's a local exported function");
    MetaAddress Address = MetaAddress::invalid();
    uint64_t Size = Symbol.st_size;

    if (IsCode) {
      Address = relocate(fromPC(Symbol.st_value));
      // TODO: record model::Function::IsDynamic = true
      model::Function *Function = nullptr;
      revng_assert(Address.isValid());
      auto It = Model->Functions().find(Address);
      if (It != Model->Functions().end()) {
        Function = &*It;
      } else {
        Function = registerFunctionEntry(Address);
        if (Function != nullptr)
          Function->Name() = Name;
      }

      if (Function != nullptr and Name.size() > 0)
        Function->ExportedNames().insert(Name.str());
    } else {
      Address = relocate(fromGeneric(Symbol.st_value));
      if (not llvm::is_contained(DataSymbols,
                                 DataSymbol{ Address, Size, Name }))
        if (IsDataObject and Size > 0)
          DataSymbols.emplace_back(Address, Size, Name);
    }
  }
}

template<typename T, bool HasAddend>
std::pair<MetaAddress, uint64_t>
ELFImporter<T, HasAddend>::ehFrameFromEhFrameHdr() {
  revng_assert(EHFrameHdrAddress);

  auto MaybeEHFrameHdr = File.getFromAddressOn(*EHFrameHdrAddress);
  if (not MaybeEHFrameHdr) {
    revng_log(ELFImporterLog,
              ".eh_frame_hdr section not available in any segment");
    return { MetaAddress::invalid(), 0 };
  }
  ArrayRef<uint8_t> EHFrameHdr = *MaybeEHFrameHdr;

  using namespace model::Architecture;
  DwarfReader<T> EHFrameHdrReader(Binary->Architecture(),
                                  EHFrameHdr,
                                  *EHFrameHdrAddress);

  uint64_t VersionNumber = EHFrameHdrReader.readNextU8();
  if (VersionNumber != 1) {
    revng_log(ELFImporterLog,
              "Unexpected version number in .eh_frame: " << VersionNumber);
    return { MetaAddress::invalid(), 0 };
  }

  // ExceptionFrameEncoding
  uint64_t ExceptionFrameEncoding = EHFrameHdrReader.readNextU8();

  // FDEsCountEncoding
  unsigned FDEsCountEncoding = EHFrameHdrReader.readNextU8();

  // LookupTableEncoding
  EHFrameHdrReader.readNextU8();

  Pointer EHFramePointer = EHFrameHdrReader.readPointer(ExceptionFrameEncoding);
  auto MaybeFDEsCount = EHFrameHdrReader.readUnsignedValue(FDEsCountEncoding);

  if (not MaybeFDEsCount) {
    revng_log(ELFImporterLog, "FDE count unavailable in .eh_frame_hdr");
    return { MetaAddress::invalid(), 0 };
  }

  MetaAddress Address = getGenericPointer(EHFramePointer);
  if (Address.isInvalid()) {
    revng_log(ELFImporterLog, "Invalid address of .eh_frame in .eh_frame_hdr");
    return { MetaAddress::invalid(), 0 };
  }

  return { Address, *MaybeFDEsCount };
}

template<typename T, bool HasAddend>
void ELFImporter<T, HasAddend>::parseEHFrame(MetaAddress EHFrameAddress,
                                             std::optional<uint64_t> FDEsCount,
                                             std::optional<uint64_t>
                                               EHFrameSize) {
  if (not FDEsCount and not EHFrameSize) {
    revng_log(ELFImporterLog, "Neither FDE count and .eh_frame size available");
    return;
  }

  // Sometimes the .eh_frame section is present but not mapped in memory. This
  // means it cannot be used at runtime, therefore we can ignore it.
  auto MaybeEHFrame = File.getFromAddressOn(EHFrameAddress);
  if (not MaybeEHFrame)
    return;
  llvm::ArrayRef<uint8_t> EHFrame = *MaybeEHFrame;

  DwarfReader<T> EHFrameReader(Model->Architecture(), EHFrame, EHFrameAddress);

  // A few fields of the CIE are used when decoding the FDE's.  This struct
  // will cache those fields we need so that we don't have to decode it
  // repeatedly for each FDE that references it.
  struct DecodedCIE {
    std::optional<uint32_t> FDEPointerEncoding;
    std::optional<uint32_t> LSDAPointerEncoding;
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
        revng_log(ELFImporterLog, ".eh_frame end was expected");
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
        // TODO: emit a diagnostic message for the user.
        revng_log(ELFImporterLog, "Unexpected version: " << Version);
        return;
      }

      // Parse a null terminated augmentation string
      SmallString<8> AugmentationString;
      for (uint8_t Char = EHFrameReader.readNextU8(); Char != 0;
           Char = EHFrameReader.readNextU8())
        AugmentationString.push_back(Char);

      // Optionally parse the EH data if the augmentation string says it's
      // there
      if (StringRef(AugmentationString).contains("eh"))
        EHFrameReader.readNextU();

      // CodeAlignmentFactor
      EHFrameReader.readULEB128();

      // DataAlignmentFactor
      EHFrameReader.readULEB128();

      // ReturnAddressRegister
      EHFrameReader.readNextU8();

      std::optional<uint64_t> AugmentationLength;
      std::optional<uint32_t> LSDAPointerEncoding;
      std::optional<uint32_t> PersonalityEncoding;
      std::optional<uint32_t> FDEPointerEncoding;
      if (!AugmentationString.empty() && AugmentationString.front() == 'z') {
        AugmentationLength = EHFrameReader.readULEB128();

        // Walk the augmentation string to get all the augmentation data.
        for (unsigned I = 1, E = AugmentationString.size(); I != E; ++I) {
          char Char = AugmentationString[I];
          switch (Char) {
          case 'e':
            if (not((I + 1) != E and AugmentationString[I + 1] == 'h')) {
              revng_log(ELFImporterLog, "Expected 'eh' in augmentation string");
              return;
            }
            break;
          case 'L':
            // This is the only information we really care about, all the
            // rest is processed just so we can get here
            if (not LSDAPointerEncoding)
              LSDAPointerEncoding = EHFrameReader.readNextU8();
            else
              revng_log(ELFImporterLog, "Duplicate LSDA encoding. Ignoroing.");

            break;
          case 'P': {
            if (PersonalityEncoding) {
              revng_log(ELFImporterLog, "Duplicate personality. Ignoring.");
              break;
            }
            PersonalityEncoding = EHFrameReader.readNextU8();
            // Personality
            Pointer Personality;
            Personality = EHFrameReader.readPointer(*PersonalityEncoding);
            auto PersonalityPtr = getCodePointer(Personality);
            logAddress(ELFImporterLog,
                       "Personality function: ",
                       PersonalityPtr);

            // Register in the model for exploration
            registerExtraCodeAddress(PersonalityPtr);
            break;
          }
          case 'R':
            if (FDEPointerEncoding) {
              revng_log(ELFImporterLog, "Duplicate FDE encoding. Ignoring.");
              break;
            }
            FDEPointerEncoding = EHFrameReader.readNextU8();
            break;
          case 'z':
            revng_log(ELFImporterLog,
                      "'z' must be first in the augmentation string");
            return;
          }
        }
      }

      // Cache this entry
      CachedCIEs[StartOffset] = { FDEPointerEncoding,
                                  LSDAPointerEncoding,
                                  AugmentationLength.has_value() };

    } else {
      // This is an FDE
      FDEIndex++;

      // The CIE pointer for an FDE is the same location as the ID which we
      // already read
      uint64_t CIEOffset = OffsetAfterLength - ID;

      // Ensure we already met this CIE
      auto CIEIt = CachedCIEs.find(CIEOffset);
      if (CIEIt == CachedCIEs.end()) {
        revng_log(ELFImporterLog,
                  "Couldn't find CIE at offset in to __eh_frame section");
        return;
      }

      // Ensure we have at least the pointer encoding
      const DecodedCIE &CIE = CIEIt->getSecond();
      if (not CIE.FDEPointerEncoding) {
        revng_log(ELFImporterLog,
                  "FDE references CIE which did not set pointer encoding");
        return;
      }

      // PCBegin
      auto PCBeginPointer = EHFrameReader.readPointer(*CIE.FDEPointerEncoding);
      MetaAddress PCBegin = getGenericPointer(PCBeginPointer);

      // PCRange
      EHFrameReader.readPointer(*CIE.FDEPointerEncoding);

      if (CIE.HasAugmentationLength)
        EHFrameReader.readULEB128();

      // Decode the LSDA if the CIE augmentation string said we should.
      if (CIE.LSDAPointerEncoding) {
        auto LSDAPointer = EHFrameReader.readPointer(*CIE.LSDAPointerEncoding);
        parseLSDA(PCBegin, getGenericPointer(LSDAPointer));
      }
    }

    // Skip all the remaining parts
    EHFrameReader.moveTo(EndOffset);
  }
}

template<typename T, bool HasAddend>
void ELFImporter<T, HasAddend>::parseLSDA(MetaAddress FDEStart,
                                          MetaAddress LSDAAddress) {
  auto MaybeLSDA = File.getFromAddressOn(LSDAAddress);
  if (not MaybeLSDA) {
    revng_log(ELFImporterLog, "LSDA not available in any segment");
    return;
  }
  llvm::ArrayRef<uint8_t> LSDA = *MaybeLSDA;

  DwarfReader<T> LSDAReader(Model->Architecture(), LSDA, LSDAAddress);

  uint32_t LandingPadBaseEncoding = LSDAReader.readNextU8();
  MetaAddress LandingPadBase = MetaAddress::invalid();
  if (LandingPadBaseEncoding != dwarf::DW_EH_PE_omit) {
    auto LandingPadBasePointer = LSDAReader.readPointer(LandingPadBaseEncoding);
    LandingPadBase = getGenericPointer(LandingPadBasePointer);
  } else {
    LandingPadBase = FDEStart;
  }

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
    MetaAddress LandingPad = this->getCodePointer(LandingPadPointer);

    // Action
    LSDAReader.readULEB128();

    if (LandingPad.isValid())
      registerExtraCodeAddress(LandingPad);
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
void ELFImporter<T, HasAddend>::registerRelocations(StringRef Name,
                                                    Elf_Rel_Array Relocations,
                                                    const FilePortion &Dynsym,
                                                    const FilePortion &Dynstr) {
  using namespace llvm::object;
  using Elf_Sym = Elf_Sym_Impl<T>;

  model::Segment *LowestSegment = nullptr;
  if (auto It = Model->Segments().begin(); It != Model->Segments().end())
    LowestSegment = &*It;

  ArrayRef<Elf_Sym> Symbols;
  if (Dynsym.isAvailable())
    Symbols = Dynsym.extractAs<Elf_Sym>();

  revng_log(ELFImporterLog, "Parsing relocations in " << Name.str());
  LoggerIndent Indent(ELFImporterLog);

  for (auto [Index, Relocation] : llvm::enumerate(Relocations)) {
    auto Type = static_cast<unsigned char>(Relocation.getType(false));
    uint64_t Addend = RelocationHelper<T, HasAddend>::getAddend(Relocation);
    MetaAddress Address = relocate(fromGeneric(Relocation.r_offset));

    StringRef SymbolName;
    unsigned char SymbolType = llvm::ELF::STT_NOTYPE;
    if (Dynsym.isAvailable() and Dynstr.isAvailable()) {
      uint32_t SymbolIndex = Relocation.getSymbol(false);
      if (not(SymbolIndex < Symbols.size())) {
        revng_log(ELFImporterLog,
                  "Invalid symbol index "
                    << SymbolIndex << ". "
                    << "Symbol count: " << Symbols.size());
      }
      const Elf_Sym &Symbol = Symbols[SymbolIndex];
      auto MaybeName = Symbol.getName(Dynstr.extractString());

      if (auto Error = MaybeName.takeError()) {
        revng_log(ELFImporterLog,
                  "Cannot access symbol name for relocation #" << Index << ": "
                                                               << Error);
        consumeError(std::move(Error));
      } else {
        SymbolName = *MaybeName;
      }

      SymbolType = Symbol.getType();
    }

    using namespace model::RelocationType;
    auto RelocationType = fromELFRelocation(Model->Architecture(), Type);

    auto RelocationName = getELFRelocationTypeName(TheBinary.ObjectFile
                                                     .getEMachine(),
                                                   Type);
    if (RelocationType == Invalid) {
      revng_log(ELFImporterLog,
                "Ignoring unknown relocation #" << Index << ": "
                                                << RelocationName);
      continue;
    }

    model::Relocation NewRelocation(Address, RelocationType, Addend);

    bool HasName = SymbolName.size() != 0;
    bool IsBaseRelative = isELFRelocationBaseRelative(Model->Architecture(),
                                                      Type);

    if (HasName and IsBaseRelative) {
      revng_log(ELFImporterLog,
                "We found a base-relative relocation #"
                  << Index << " (" << RelocationName
                  << ") associated to a symbol, ignoring.");
    } else if (not HasName and not IsBaseRelative) {
      if (ELFImporterLog.isEnabled()) {
        ELFImporterLog << "We found a non-base-relative relocation #" << Index
                       << " (" << RelocationName
                       << ") not associated to a symbol, ignoring." << DoLog;
      }
    } else if (HasName) {
      // Symbol-relative relocation
      if (SymbolType == ELF::STT_FUNC) {
        auto It = Model->ImportedDynamicFunctions().find(SymbolName.str());
        if (It != Model->ImportedDynamicFunctions().end()) {
          auto &Relocations = It->Relocations();
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
        LowestSegment->Relocations().insert(NewRelocation);
      } else {
        revng_log(ELFImporterLog,
                  "Found a base-relative relocation #" << Index
                                                       << ", but no segment is "
                                                          "available! "
                                                          "Ignoring.");
      }
    }
  }
}

static std::unique_ptr<ELFImporterBase>
createELFImporter(TupleTree<model::Binary> &M,
                  const ELFBinary &TheBinary,
                  bool IsLittleEndian,
                  uint64_t PointerSize,
                  bool HasRelocationAddend,
                  uint64_t BaseAddress) {
  if (PointerSize != 4 and PointerSize != 8)
    revng_abort("Unexpected address size");

  if (TheBinary.ObjectFile.getEType() != ELF::ET_DYN)
    BaseAddress = 0;

  // In the case of MIPS architecture, we handle some specific import
  // as a part of a separate derived (from ELFImporter) class.
  // TODO: Investigate other architectures as well.
  bool IsMIPS = (M->Architecture() == model::Architecture::mips
                 or M->Architecture() == model::Architecture::mipsel);
  bool Is32Bit = PointerSize == 4;

  auto Maker = [&]<bool Is32Bit,
                   bool IsLittleEndian,
                   bool HasRelocationAddend,
                   bool IsMIPS>() -> std::unique_ptr<ELFImporterBase> {
    using ELF32 = std::conditional_t<IsLittleEndian, ELF32LE, ELF32BE>;
    using ELF64 = std::conditional_t<IsLittleEndian, ELF64LE, ELF64BE>;
    using ELFType = std::conditional_t<Is32Bit, ELF32, ELF64>;

    using EI = ELFImporter<ELFType, HasRelocationAddend>;
    using MEI = MIPSELFImporter<ELFType, HasRelocationAddend>;
    using ImporterType = std::conditional_t<IsMIPS, MEI, EI>;
    return std::make_unique<ImporterType>(M, TheBinary, BaseAddress);
  };
  return compile_time::invokeCombination(Maker,
                                         Is32Bit,
                                         IsLittleEndian,
                                         HasRelocationAddend,
                                         IsMIPS);
}

Error importELF(TupleTree<model::Binary> &Model,
                const ELFBinary &TheBinary,
                const ImporterOptions &Options) {
  // In the case of MIPS architecture, we handle some specific import
  // as a part of a separate derived (from ELFImporter) class.
  // TODO: Investigate other architectures as well.
  bool IsMIPS = (Model->Architecture() == model::Architecture::mips
                 or Model->Architecture() == model::Architecture::mipsel);

  using namespace model::Architecture;
  bool IsLittleEndian = isLittleEndian(Model->Architecture());
  uint64_t PointerSize = getPointerSize(Model->Architecture());
  bool HasRelocationAddend = hasELFRelocationAddend(Model->Architecture());

  auto Importer = createELFImporter(Model,
                                    TheBinary,
                                    IsLittleEndian,
                                    PointerSize,
                                    HasRelocationAddend,
                                    Options.BaseAddress);
  return Importer->import(Options);
}
