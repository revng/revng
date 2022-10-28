/// \file PECOFF.cpp
/// \brief

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/Object/COFF.h"
#include "llvm/Object/ObjectFile.h"

#include "revng/ABI/DefaultFunctionPrototype.h"
#include "revng/Model/Importer/Binary/BinaryImporterHelper.h"
#include "revng/Model/Binary.h"
#include "revng/Model/IRHelpers.h"
#include "revng/Model/Importer/DebugInfo/PDBImporter.h"
#include "revng/Support/Debug.h"

#include "Importers.h"

using namespace llvm;
using namespace llvm::object;

static Logger<> Log("pecoff-importer");

class PECOFFImporter : public BinaryImporterHelper {
private:
  TupleTree<model::Binary> &Model;
  const object::COFFObjectFile &TheBinary;
  MetaAddress ImageBase = MetaAddress::invalid();

public:
  PECOFFImporter(TupleTree<model::Binary> &Model,
                 const object::COFFObjectFile &TheBinary) :
    Model(Model), TheBinary(TheBinary) {}

  Error import();

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
  void
  recordDelayImportedFunctions(DelayDirectoryRef &I, ImportedSymbolRange Range);
};

Error PECOFFImporter::parseSectionsHeaders() {
  using namespace model;

  revng_assert(Model->Architecture != Architecture::Invalid);
  Architecture = Model->Architecture;

  auto PointerSize = Architecture::getPointerSize(Architecture);
  bool IsLittleEndian = Architecture::isLittleEndian(Architecture);

  if ((PointerSize != 4 and PointerSize != 8) or not IsLittleEndian)
    return createError("Only 32/64-bit little endian COFF files are supported");

  const object::pe32_header *PE32Header = TheBinary.getPE32Header();

  if (PE32Header) {
    // TODO: ImageBase should aligned to 4kb pages, should we check that?
    ImageBase = fromPC(PE32Header->ImageBase);

    Model->EntryPoint = ImageBase + u64(PE32Header->AddressOfEntryPoint);
  } else {
    const object::pe32plus_header *PE32PlusHeader = TheBinary
                                                      .getPE32PlusHeader();
    if (not PE32PlusHeader)
      return createError("Invalid PE Header");

    // PE32+ Header
    ImageBase = fromPC(PE32PlusHeader->ImageBase);
    Model->EntryPoint = ImageBase + u64(PE32PlusHeader->AddressOfEntryPoint);
  }

  // Read sections
  for (const SectionRef &SecRef : TheBinary.sections()) {
    unsigned Id = TheBinary.getSectionID(SecRef);
    Expected<const object::coff_section *> SecOrErr = TheBinary.getSection(Id);
    if (not SecOrErr) {
      revng_log(Log,
                "Error in section with ID " << Id << ": "
                                            << SecOrErr.takeError());
      continue;
    }
    const object::coff_section *CoffRef = *SecOrErr;

    MetaAddress Start = ImageBase + u64(CoffRef->VirtualAddress);
    Segment Segment({ Start, u64(CoffRef->VirtualSize) });

    Segment.StartOffset = CoffRef->PointerToRawData;

    // VirtualSize might be larger than SizeOfRawData (extra data at the end of
    // the section) or viceversa (data mapped in memory but not present in
    // memory, e.g., .bss)
    Segment.FileSize = CoffRef->SizeOfRawData;

    // Since it is possible that the file size is greater than VirtualSize
    // because SizeOfRawData is rounded, but VirtualSize is not, we work it
    // around here by using maximum of these two values for the VirtSize.
    if (Segment.FileSize > Segment.VirtualSize)
      Segment.VirtualSize = Segment.FileSize;

    Segment.IsReadable = CoffRef->Characteristics & COFF::IMAGE_SCN_MEM_READ;
    Segment.IsWriteable = CoffRef->Characteristics & COFF::IMAGE_SCN_MEM_WRITE;
    Segment.IsExecutable = CoffRef->Characteristics
                           & COFF::IMAGE_SCN_MEM_EXECUTE;

    model::TypePath StructPath = createEmptyStruct(*Model, Segment.VirtualSize);
    Segment.Type = model::QualifiedType(std::move(StructPath), {});

    // NOTE: Unlike ELF, PE/COFF does not have segments. Instead, it has
    // sections only. All the raw data in a section must be loaded
    // contiguously. Segments just map file range to virtual address space
    // range.
    auto SectionStart = ImageBase + u64(CoffRef->VirtualAddress);
    uint64_t Size = u64(CoffRef->VirtualSize);
    auto SectionEnd = SectionStart + Size;

    if (SectionStart.isValid() and SectionEnd.isValid()
        and SectionStart.addressLowerThan(SectionEnd)) {
      model::Section NewSection(SectionStart, Size);
      if (auto SectionName = TheBinary.getSectionName(CoffRef))
        NewSection.Name = SectionName->str();
      NewSection.ContainsCode = Segment.IsExecutable;
      revng_assert(NewSection.verify(true));
      Segment.Sections.insert(std::move(NewSection));
    } else {
      revng_log(Log, "Found an invalid section");
    }

    Segment.verify(true);
    Model->Segments.insert(std::move(Segment));
  }

  return Error::success();
}

void PECOFFImporter::parseSymbols() {
  for (auto Sym : TheBinary.symbols()) {
    COFFSymbolRef Symbol = TheBinary.getCOFFSymbol(Sym);

    if (!Symbol.isFunctionDefinition())
      continue;

    Expected<StringRef> NameOrErr = TheBinary.getSymbolName(Symbol);
    if (!NameOrErr) {
      revng_log(Log, "Found static symbol without a name.");
      continue;
    }

    // Relocate the symbol.
    MetaAddress Address = ImageBase + Symbol.getValue();
    if (Model->Functions.count(Address))
      continue;

    model::Function &Function = Model->Functions[Address];
    Function.OriginalName = *NameOrErr;
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
    if (Sym.empty() or Model->ImportedDynamicFunctions.count(Sym.str()))
      continue;

    // NOTE: This address will occur in the .text section as a target of a jump.
    // Once we have the address of the entry within .idata, we can access
    // the information about symbol.
    auto PointerSize = getPointerSize(Model->Architecture);
    MetaAddress AddressOfImportEntry = ImageBase + u64(ImportAddressTableEntry)
                                       + u64(Index * PointerSize);

    // Lets make a Relocation.
    using namespace model::RelocationType;
    auto RelocationType = formCOFFRelocation(Model->Architecture);
    model::Relocation NewRelocation(AddressOfImportEntry, RelocationType);

    auto It = Model->ImportedDynamicFunctions.insert(Sym.str()).first;
    revng_assert(NewRelocation.verify(true));
    It->Relocations.insert(NewRelocation);
    ++Index;
  }
}

void PECOFFImporter::parseImportedSymbols() {
  for (const ImportDirectoryEntryRef &I : TheBinary.import_directories()) {
    StringRef Name;
    if (Error E = I.getName(Name)) {
      revng_log(Log, "Found an imported symbol without a name.");
      continue;
    }

    // Let's find symbols of the dll associated with Name.

    uint32_t ImportLookupTableEntry;
    if (Error E = I.getImportLookupTableRVA(ImportLookupTableEntry)) {
      revng_log(Log, "No ImportLookupTableRVA found for an import");
      continue;
    }

    uint32_t ImportAddressTableEntry;
    if (Error E = I.getImportAddressTableRVA(ImportAddressTableEntry)) {
      revng_log(Log, "No ImportAddressTableRVA found for an import");
      continue;
    }

    if (not Model->ImportedLibraries.insert(Name.str()).second)
      continue;

    // The import lookup table can be missing with certain older linkers, so
    // fall back to the import address table in that case.
    if (ImportLookupTableEntry) {
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
    if (Sym.empty() or Model->ImportedDynamicFunctions.count(Sym.str()))
      continue;

    MetaAddress AddressOfDelayImportEntry = ImageBase + u64(Addr);

    // Lets make Relocation.
    using namespace model::RelocationType;
    auto RelocationType = formCOFFRelocation(Model->Architecture);
    model::Relocation NewRelocation(AddressOfDelayImportEntry, RelocationType);
    auto NewIt = Model->ImportedDynamicFunctions.insert(Sym.str()).first;
    revng_assert(NewRelocation.verify(true));
    NewIt->Relocations.insert(NewRelocation);
  }
}

void PECOFFImporter::parseDelayImportedSymbols() {
  for (DelayDirectoryRef &I : TheBinary.delay_import_directories()) {
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

    if (not Model->ImportedLibraries.insert(Name.str()).second)
      continue;

    recordDelayImportedFunctions(I, I.imported_symbols());
  }
}

Error PECOFFImporter::import() {
  if (Error E = parseSectionsHeaders())
    return E;

  // Parse the symbol table.
  parseSymbols();

  // Parse dynamic symbol table.
  parseImportedSymbols();

  // Parse delay dynamic symbol table (similar to ELF's symbols used for lazy
  // linking).
  parseDelayImportedSymbols();

  if (Model->DefaultABI == model::ABI::Invalid)
    Model->DefaultABI = model::ABI::getDefaultMicrosoftABI(Model->Architecture);

  // Create a default prototype.
  Model->DefaultPrototype = abi::registerDefaultFunctionPrototype(*Model);

  PDBImporter PDBI(Model, ImageBase);
  PDBI.import(TheBinary);

  return Error::success();
}

Error importPECOFF(TupleTree<model::Binary> &Model,
                   const object::COFFObjectFile &TheBinary,
                   uint64_t PreferredBaseAddress) {
  // TODO: use PreferredBaseAddress if PIC
  (void) PreferredBaseAddress;

  PECOFFImporter Importer(Model, TheBinary);
  return Importer.import();
}
