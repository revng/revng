//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/Object/COFF.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MathExtras.h"

#include "revng/ABI/DefaultFunctionPrototype.h"
#include "revng/Model/Binary.h"
#include "revng/Model/IRHelpers.h"
#include "revng/Model/Importer/Binary/BinaryDescriptor.h"
#include "revng/Model/Importer/Binary/BinaryImporterHelper.h"
#include "revng/Model/Importer/Binary/Options.h"
#include "revng/Model/Importer/DebugInfo/PDBImporter.h"
#include "revng/Model/Pass/AllPasses.h"
#include "revng/Support/Debug.h"
#include "revng/Support/Error.h"
#include "revng/Support/MetaAddress.h"

#include "CrossModelFindTypeHelper.h"
#include "Importers.h"

using namespace llvm;
using namespace llvm::object;

static Logger Log("pecoff-importer");

// Force using a specific PDB.
static llvm::cl::opt<std::string> UsePDB("use-pdb",
                                         llvm::cl::desc("Path to the PDB."),
                                         llvm::cl::cat(MainCategory));

using PELDDTree = std::map<std::string, std::vector<std::string>>;

class PECOFFImporter : public BinaryImporterHelper {
private:
  TupleTree<model::Binary> &Model;
  const COFFBinary &TheBinary;
  llvm::StringRef BinaryPath;
  MetaAddress ImageBase = MetaAddress::invalid();

public:
  PECOFFImporter(TupleTree<model::Binary> &Model,
                 const COFFBinary &TheBinary,
                 uint64_t BaseAddress) :
    BinaryImporterHelper(Model, BaseAddress, Log),
    Model(Model),
    TheBinary(TheBinary) {
    revng_log(Log, "Creating binary importer helper");
  }

  Error import(const ImporterOptions &Options);

private:
  Error parseSectionsHeaders();
  /// Parse static symbols from the file.
  void parseSymbols();

  /// Parse dynamic symbols from the file.
  void parseImportedSymbols();
  using ImportedSymbolRange = iterator_range<imported_symbol_iterator>;
  void recordImportedFunctions(ImportedSymbolRange Range,
                               uint32_t ImportAddressTableEntry);
  /// Parse delay dynamic symbols from the file.
  void parseDelayImportedSymbols();

  using DelayDirectoryRef = const DelayImportDirectoryEntryRef;
  void recordDelayImportedFunctions(DelayDirectoryRef &I,
                                    ImportedSymbolRange Range);
};

Error PECOFFImporter::parseSectionsHeaders() {
  using namespace model;

  revng_assert(Model->Architecture() != Architecture::Invalid);

  uint64_t PointerSize = Architecture::getPointerSize(Model->Architecture());
  bool IsLittleEndian = Architecture::isLittleEndian(Model->Architecture());

  if ((PointerSize != 4 and PointerSize != 8) or not IsLittleEndian) {
    return revng::createError("Only 32/64-bit little endian COFF files are "
                              "supported");
  }

  const object::pe32_header *PE32Header = TheBinary.ObjectFile.getPE32Header();

  // Identify ImageBase
  if (PE32Header) {
    // TODO: ImageBase should aligned to 4kb pages, should we check that?
    ImageBase = fromGeneric(PE32Header->ImageBase);
  } else {
    const pe32plus_header *PE32PlusHeader = TheBinary.ObjectFile
                                              .getPE32PlusHeader();
    if (not PE32PlusHeader)
      return revng::createError("Invalid PE Header");

    // PE32+ Header
    ImageBase = fromGeneric(PE32PlusHeader->ImageBase);
  }

  // Read sections
  for (const SectionRef &SecRef : TheBinary.ObjectFile.sections()) {
    unsigned Id = TheBinary.ObjectFile.getSectionID(SecRef);
    Expected<const object::coff_section *> MaybeSection = TheBinary.ObjectFile
                                                            .getSection(Id);
    if (auto Error = MaybeSection.takeError()) {
      // TODO: emit a diagnostic message for the user.
      revng_log(Log, "Error in section with ID " << Id << ": " << Error);
      consumeError(std::move(Error));
      continue;
    }
    const object::coff_section *CoffRef = *MaybeSection;

    MetaAddress Start = ImageBase + u64(CoffRef->VirtualAddress);
    Segment Segment({ Start, u64(CoffRef->VirtualSize) });

    // TODO: do the following unconditionally once the old pipeline has been
    //       dropped.
    if (TheBinary.Reference.isValid())
      Segment.Binary() = TheBinary.Reference;
    Segment.StartOffset() = CoffRef->PointerToRawData;

    // VirtualSize might be larger than SizeOfRawData (extra data at the end of
    // the section) or vice-versa (data mapped in memory but not present in
    // memory, e.g., .bss)
    Segment.FileSize() = CoffRef->SizeOfRawData;

    // Since it is possible that the file size is greater than VirtualSize
    // because SizeOfRawData is rounded, but VirtualSize is not, we work it
    // around here by using maximum of these two values for the VirtSize.
    if (Segment.FileSize() > Segment.VirtualSize())
      Segment.VirtualSize() = Segment.FileSize();

    Segment.IsReadable() = CoffRef->Characteristics & COFF::IMAGE_SCN_MEM_READ;
    Segment.IsWriteable() = CoffRef->Characteristics
                            & COFF::IMAGE_SCN_MEM_WRITE;
    Segment.IsExecutable() = CoffRef->Characteristics
                             & COFF::IMAGE_SCN_MEM_EXECUTE;

    // TODO: replace the following with `populateSegmentTypeStruct`, when
    // symbol table and dynamic symbol table parsing is finalized
    auto &&[Struct, Type] = Model->makeStructDefinition(Segment.VirtualSize());
    Struct.CanContainCode() = Segment.IsExecutable();
    Segment.Type() = std::move(Type);

    Segment.verify(true);
    Model->Segments().insert(std::move(Segment));
  }

  processSegments();

  // Identify EntryPoint
  MetaAddress EntryPoint;
  if (PE32Header) {
    if (PE32Header->AddressOfEntryPoint != 0) {
      EntryPoint = toPC(ImageBase + u64(PE32Header->AddressOfEntryPoint));
    }
  } else {
    const pe32plus_header *PE32PlusHeader = TheBinary.ObjectFile
                                              .getPE32PlusHeader();
    revng_assert(PE32PlusHeader);

    // PE32+ Header
    if (PE32PlusHeader->AddressOfEntryPoint != 0) {
      EntryPoint = toPC(ImageBase + u64(PE32PlusHeader->AddressOfEntryPoint));
    }
  }

  if (EntryPoint.isValid())
    setEntryPoint(EntryPoint);

  return Error::success();
}

void PECOFFImporter::parseSymbols() {
  for (auto Sym : TheBinary.ObjectFile.symbols()) {
    COFFSymbolRef Symbol = TheBinary.ObjectFile.getCOFFSymbol(Sym);

    if (!Symbol.isFunctionDefinition())
      continue;

    Expected<StringRef> MaybeName = TheBinary.ObjectFile.getSymbolName(Symbol);
    if (auto Error = MaybeName.takeError()) {
      revng_log(Log, "Found static symbol without a name.");
      consumeError(std::move(Error));
      continue;
    }

    // Relocate the symbol.
    MetaAddress Address = toPC(ImageBase + Symbol.getValue());
    if (Model->Functions().contains(Address))
      continue;

    if (auto *Function = registerFunctionEntry(Address))
      Function->Name() = *MaybeName;
  }
}

void PECOFFImporter::recordImportedFunctions(ImportedSymbolRange Range,
                                             uint32_t ImportAddressTableEntry) {

  // Index of entries within import table.
  uint32_t Index = 0;
  for (const ImportedSymbolRef &I : Range) {
    StringRef Sym;
    if (Error E = I.getSymbolName(Sym)) {
      revng_log(Log, "Found an imported symbol without a name.");
      continue;
    }

    // TODO: We may face some old linkers that use ordinal bits only
    // so consider this info then.
    uint16_t Ordinal;
    if (Error E = I.getOrdinal(Ordinal)) {
      revng_log(Log, "Found an imported symbol without an ordinal.");
      continue;
    }

    // Dynamic functions must have a name, so skip those without it.
    // TODO: handle imports by ordinal
    if (Sym.empty() or Model->ImportedDynamicFunctions().contains(Sym.str()))
      continue;

    // NOTE: This address will occur in the .text section as a target of a jump.
    // Once we have the address of the entry within .idata, we can access
    // the information about symbol.
    uint64_t PointerSize = getPointerSize(Model->Architecture());
    MetaAddress AddressOfImportEntry = ImageBase + u64(ImportAddressTableEntry)
                                       + u64(Index * PointerSize);

    // Lets make a Relocation.
    using namespace model::RelocationType;
    auto RelocationType = formCOFFRelocation(Model->Architecture());
    model::Relocation NewRelocation(AddressOfImportEntry, RelocationType);

    auto It = Model->ImportedDynamicFunctions().insert(Sym.str()).first;
    revng_assert(NewRelocation.verify(true));
    It->Relocations().insert(NewRelocation);
    ++Index;
  }
}

void PECOFFImporter::parseImportedSymbols() {
  for (const ImportDirectoryEntryRef &I :
       TheBinary.ObjectFile.import_directories()) {
    StringRef Name;
    if (Error E = I.getName(Name)) {
      revng_log(Log, "Found an imported library without a name.");
      continue;
    }

    // Let's find symbols of the dll associated with Name.

    // In PE/COFF you can have two tables for imported functions: the
    // ImportLookupTable (aka ILT) and the ImportAddressTable (aka IAT). On disk
    // they are supposed to be the same. At run-time the IAT will be patched
    // with the actual addresses.
    //
    // Exception: the IAT might be "bound", i.e., pre-filled. The loader checks
    // at run-time if the bound IAT is valid via a timestamp. If it isn't uses
    // the ILT. One could debate which one is more reliable, we choose to first
    // use the ILT, if absent, we use the IAT.

    uint32_t ImportLookupTableEntry = 0;
    bool HasImportLookupTableEntry = true;
    if (Error E = I.getImportLookupTableRVA(ImportLookupTableEntry)) {
      revng_log(Log, "No ImportLookupTableRVA found for an import");
      HasImportLookupTableEntry = false;
    }

    uint32_t ImportAddressTableEntry;
    if (Error E = I.getImportAddressTableRVA(ImportAddressTableEntry)) {
      revng_log(Log, "No ImportAddressTableRVA found for an import");
      continue;
    }

    if (not Model->ImportedLibraries().insert(Name.str()).second) {
      revng_log(Log, "Library " << Name.str() << " already imported, skipping");
      continue;
    }

    // The import lookup table can be missing with certain older linkers, so
    // fall back to the import address table in that case.
    if (HasImportLookupTableEntry) {
      recordImportedFunctions(I.lookup_table_symbols(),
                              ImportAddressTableEntry);
    } else {
      recordImportedFunctions(I.imported_symbols(), ImportAddressTableEntry);
    }
  }
}

void PECOFFImporter::recordDelayImportedFunctions(DelayDirectoryRef &I,
                                                  ImportedSymbolRange Range) {
  // Index of entries within import table.
  uint32_t Index = 0;
  for (const ImportedSymbolRef &S : Range) {
    StringRef Sym;
    if (Error E = S.getSymbolName(Sym)) {
      revng_log(Log, "Found a delay imported symbol without a name.");
      continue;
    }

    // TODO: We may face some old linkers that use ordinal bits only
    // so consider this info then.
    uint16_t Ordinal;
    if (Error E = S.getOrdinal(Ordinal)) {
      revng_log(Log, "Found a delay imported symbol without an ordinal.");
      continue;
    }

    uint64_t Addr;
    if (Error E = I.getImportAddress(Index++, Addr)) {
      revng_log(Log, "Found a delay imported symbol without an address.");
      continue;
    }

    // Dynamic functions must have a name, so skip those without it.
    // TODO: handle imports by ordinal
    if (Sym.empty() or Model->ImportedDynamicFunctions().contains(Sym.str()))
      continue;

    MetaAddress AddressOfDelayImportEntry = ImageBase + u64(Addr);

    // Lets make Relocation.
    using namespace model::RelocationType;
    auto RelocationType = formCOFFRelocation(Model->Architecture());
    model::Relocation NewRelocation(AddressOfDelayImportEntry, RelocationType);
    auto NewIt = Model->ImportedDynamicFunctions().insert(Sym.str()).first;
    revng_assert(NewRelocation.verify(true));
    NewIt->Relocations().insert(NewRelocation);
  }
}

void PECOFFImporter::parseDelayImportedSymbols() {
  for (DelayDirectoryRef &I : TheBinary.ObjectFile.delay_import_directories()) {
    StringRef Name;
    if (Error E = I.getName(Name)) {
      revng_log(Log, "No name of a delay imported dll.");
      continue;
    }

    const delay_import_directory_table_entry *Table;
    if (Error E = I.getDelayImportTable(Table)) {
      revng_log(Log, "No delay import table found for a dll.");
      continue;
    }

    if (not Model->ImportedLibraries().insert(Name.str()).second)
      continue;

    recordDelayImportedFunctions(I, I.imported_symbols());
  }
}

Error PECOFFImporter::import(const ImporterOptions &Options) {
  if (Error E = parseSectionsHeaders())
    return E;

  // Parse the symbol table.
  parseSymbols();

  // Parse dynamic symbol table.
  parseImportedSymbols();

  // Parse delay dynamic symbol table (similar to ELF's symbols used for lazy
  // linking).
  parseDelayImportedSymbols();

  // Set default ABI
  if (Model->DefaultABI() == model::ABI::Invalid) {
    revng_assert(Model->Architecture() != model::Architecture::Invalid);
    if (auto ABI = model::ABI::getDefaultForPECOFF(Model->Architecture())) {
      Model->DefaultABI() = ABI.value();
    } else {
      auto ArchName = model::Architecture::getName(Model->Architecture()).str();
      return revng::createError("Unsupported architecture for PECOFF: "
                                + ArchName);
    }
  }

  // Create a default prototype.
  Model->DefaultPrototype() = abi::registerDefaultFunctionPrototype(*Model);

  if (Options.DebugInfo != DebugInfoLevel::No) {
    revng_log(Log, "Importing PDB");
    LoggerIndent Indent(Log);

    // Consider the --use-pdb argument
    if (not UsePDB.empty()) {
      auto ImportLogger = importLogger(TheBinary.canonicalPath());
      PDBImporter PDBI(Model, ImageBase, std::nullopt);
      PDBI.importPDB(UsePDB, Options);
    }

    // Identify dependencies
    std::optional<LDDTree>
      MaybeDependencies = identifyDependencies(TheBinary.ObjectFile,
                                               TheBinary.canonicalPath());

    if (MaybeDependencies.has_value()) {
      // Import debug info
      {
        auto ImportLogger = importLogger(TheBinary.canonicalPath());
        PDBImporter PDBI(Model, ImageBase, std::nullopt);
        revng_assert(MaybeDependencies->Root != nullptr);
        PDBI.import(MaybeDependencies->Root, TheBinary, Options);
      }

      // Now we try to find missing types in the dependencies
      findMissingTypes<COFFBinary, PDBImporter>(*MaybeDependencies,
                                                Options,
                                                Logger,
                                                Model);
    } else {
      revng_log(Log, "Couldn't find an appropriate rootfs");
    }
  }

  model::flattenPrimitiveTypedefs(Model);
  model::deduplicateCollidingNames(Model);
  return Error::success();
}

Error importPECOFF(TupleTree<model::Binary> &Model,
                   const COFFBinary &TheBinary,
                   const ImporterOptions &Options) {
  PECOFFImporter Importer(Model, TheBinary, Options.BaseAddress);
  return Importer.import(Options);
}
