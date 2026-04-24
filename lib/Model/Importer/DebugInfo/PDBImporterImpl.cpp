//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/DebugInfo/CodeView/CVSymbolVisitor.h"
#include "llvm/DebugInfo/CodeView/CVTypeVisitor.h"
#include "llvm/DebugInfo/CodeView/SymbolDeserializer.h"
#include "llvm/DebugInfo/CodeView/SymbolVisitorCallbackPipeline.h"
#include "llvm/DebugInfo/CodeView/SymbolVisitorCallbacks.h"
#include "llvm/DebugInfo/PDB/Native/DbiStream.h"
#include "llvm/DebugInfo/PDB/Native/TpiStream.h"

#include "revng/Model/CABIFunctionDefinition.h"
#include "revng/Model/Pass/AllPasses.h"
#include "revng/Model/PrimitiveType.h"
#include "revng/Model/Processing.h"
#include "revng/Model/TypedefDefinition.h"
#include "revng/Support/OverflowSafeInt.h"

#include "PDBImporterImpl.h"
#include "PDBTypeResolver.h"

using namespace llvm;
using namespace llvm::codeview;
using namespace llvm::pdb;

void PDBImporterImpl::populateSymbolsWithTypes() {
  revng_log(Log, "populateSymbolsWithTypes");
  LoggerIndent Indent(Log);

  for (ProcSym &ProcedureSymbol : Symbols.ProcSymRecords)
    handleProcedureSymbol(ProcedureSymbol);

  for (auto &[Index, Function] : FunctionRecords.FuncIdRecords) {
    auto Name = Function.getName().str();

    // We use FuncIdRecords only to harvest the type of dynamic symbols.
    // The list also contains local symbols, but we already know about them from
    // ProcSymRecords, so we skip them to avoid defining both a local and
    // dynamic symbol for a certain string.
    if (LocalSymbolNames.contains(Name))
      continue;

    if (not Model->ImportedDynamicFunctions().contains(Name)) {
      revng_log(Log,
                "Creating dynamic function " << Name << " due to FuncIdRecord "
                                             << toString(Index));
    }

    auto &DynamicFunction = Model->ImportedDynamicFunctions()[Name];
    if (auto *Type = tryGetType(Function.getFunctionType()))
      DynamicFunction.Prototype() = Type->copy();
  }
}

class TypeRecordsCollector : public llvm::codeview::TypeVisitorCallbacks {
public:
  using TypeIndex = llvm::codeview::TypeIndex;
  using CVType = llvm::codeview::CVType;

private:
  Records &Records;
  TypeIndex CurrentTypeIndex = TypeIndex::None();

public:
  TypeRecordsCollector(struct Records &Records, TypeIndex FirstTypeIndex) :
    Records(Records), CurrentTypeIndex(FirstTypeIndex) {}

public:
  llvm::Error visitTypeBegin(llvm::codeview::CVType &Record) override {
    revng_log(Log, "visitType " << toString(CurrentTypeIndex));
    Log.indent();
    return llvm::Error::success();
  }

  llvm::Error visitTypeBegin(CVType &Record, TypeIndex TI) override {
    revng_abort("We don't expect this to be called");
  }

  llvm::Error visitTypeEnd(CVType &Record) override {
    Log.unindent();
    ++CurrentTypeIndex;
    return llvm::Error::success();
  }

#define VISIT_RECORD(Name)                                                   \
  llvm::Error visitKnownRecord(CVType &Record, llvm::codeview::Name &Object) \
    override {                                                               \
    revng_log(Log, "Registering " #Name);                                    \
    registerObject(Records.Name##s, std::move(Object));                      \
    return llvm::Error::success();                                           \
  }

  VISIT_RECORD(ClassRecord);
  VISIT_RECORD(EnumRecord);
  VISIT_RECORD(ProcedureRecord);
  VISIT_RECORD(MemberFunctionRecord);
  VISIT_RECORD(UnionRecord);
  VISIT_RECORD(ArgListRecord);
  VISIT_RECORD(PointerRecord);
  VISIT_RECORD(ModifierRecord);
  VISIT_RECORD(ArrayRecord);
  VISIT_RECORD(BitFieldRecord);
  VISIT_RECORD(AliasRecord);

  VISIT_RECORD(FuncIdRecord);

#undef VISIT_RECORD

  llvm::Error
  visitKnownRecord(CVType &Record,
                   llvm::codeview::FieldListRecord &FieldList) override {
    revng_log(Log,
              "Visiting FieldListRecord (size: " << FieldList.Data.size()
                                                 << ")");
    LoggerIndent Indent(Log);

    // We don't know what's going to contain
    Records.EnumeratorRecords.changeKey(CurrentTypeIndex);
    Records.DataMemberRecords.changeKey(CurrentTypeIndex);

    if (auto Error = visitMemberRecordStream(FieldList.Data, *this)) {
      std::string Message = llvm::toString(std::move(Error));
      revng_log(Log, "Warning: failed to visit FieldListRecord: " << Message);
    }

    return llvm::Error::success();
  }

#define VISIT_MEMBER(Name)                                              \
  llvm::Error visitKnownMember(llvm::codeview::CVMemberRecord &Record,  \
                               llvm::codeview::Name &Object) override { \
    revng_log(Log, "Registering " #Name);                               \
    Records.Name##s.pushBack(std::move(Object));                        \
    return llvm::Error::success();                                      \
  }

  VISIT_MEMBER(EnumeratorRecord);
  VISIT_MEMBER(DataMemberRecord);

#undef VISIT_MEMBER

private:
  template<typename T>
  void registerObject(Records::TypeList<T> &ObjectList, T &&Object) {
    if (Records.RecordsByIndex.count(CurrentTypeIndex) != 0) {
      revng_log(Log,
                "Warning: type with index " << toString(CurrentTypeIndex)
                                            << " already registered, ignoring");
      return;
    }

    ObjectList.push_back({ CurrentTypeIndex, std::move(Object) });
    Records.RecordsByIndex[CurrentTypeIndex] = &ObjectList.back().second;
  }
};

class SymbolsCollector : public llvm::codeview::SymbolVisitorCallbacks {
public:
  using CVSymbol = llvm::codeview::CVSymbol;

private:
  Symbols &Symbols;

public:
  SymbolsCollector(struct Symbols &Symbols) : Symbols(Symbols) {}

public:
  llvm::Error visitKnownRecord(CVSymbol &Record,
                               llvm::codeview::Compile2Sym &Object) override {
    Symbols.Compile2Records.push_back(std::move(Object));
    return llvm::Error::success();
  }

  llvm::Error visitKnownRecord(CVSymbol &Record,
                               llvm::codeview::Compile3Sym &Object) override {
    Symbols.Compile3Records.push_back(std::move(Object));
    return llvm::Error::success();
  }

  llvm::Error visitKnownRecord(CVSymbol &Record,
                               llvm::codeview::UDTSym &Object) override {
    Symbols.UDTSymRecords.push_back(std::move(Object));
    return llvm::Error::success();
  }

  llvm::Error visitKnownRecord(CVSymbol &Record,
                               llvm::codeview::ProcSym &Object) override {
    Symbols.ProcSymRecords.push_back(std::move(Object));
    return llvm::Error::success();
  }
};

void PDBImporterImpl::collectRecords() {
  revng_log(Log, "Running collectRecords");
  LoggerIndent Indent(Log);

  auto &PDBFile = *Importer.getPDBFile();

  auto VisitStream = [](Expected<TpiStream &> MaybeStream, Records &Records) {
    if (not MaybeStream) {
      std::string Message = llvm::toString(MaybeStream.takeError());
      revng_log(Log, "Failed to get stream: " << Message);
      return;
    }

    TypeRecordsCollector Collector(Records,
                                   TypeIndex(MaybeStream->TypeIndexBegin()));

    bool HadError = false;
    auto Result = visitTypeStream(MaybeStream->types(&HadError), Collector);
    if (Result) {
      std::string Message = llvm::toString(std::move(Result));
      revng_log(Log, "Failure visiting the type stream: " << Message);
      return;
    }

    if (HadError) {
      revng_log(Log, "TPISream::types reported an error");
      return;
    }
  };

  {
    revng_log(Log, "Visiting Tpi stream");
    LoggerIndent Indent(Log);
    VisitStream(PDBFile.getPDBTpiStream(), TypeRecords);
  }
  {
    revng_log(Log, "Visiting Ipi stream");
    LoggerIndent Indent(Log);
    VisitStream(PDBFile.getPDBIpiStream(), FunctionRecords);
  }

  TypeRecords.finalize();

  SymbolsCollector CompileSymbolsVisitor(Symbols);
  visitSymbols(CompileSymbolsVisitor);
}

static model::Architecture::Values
toModelArchitecture(llvm::codeview::CPUType Type) {
  using namespace llvm::codeview;
  switch (Type) {
  case CPUType::Intel8080:
  case CPUType::Intel8086:
  case CPUType::Intel80286:
  case CPUType::Intel80386:
  case CPUType::Intel80486:
  case CPUType::Pentium:
  case CPUType::PentiumPro:
  case CPUType::Pentium3:
    return model::Architecture::x86;

  case CPUType::X64:
    return model::Architecture::x86_64;

  case CPUType::ARM3:
  case CPUType::ARM4:
  case CPUType::ARM4T:
  case CPUType::ARM5:
  case CPUType::ARM5T:
  case CPUType::ARM6:
  case CPUType::ARM_XMAC:
  case CPUType::ARM_WMMX:
  case CPUType::ARM7:
  case CPUType::Thumb:
  case CPUType::ARMNT:
    return model::Architecture::arm;

  case CPUType::ARM64:
  case CPUType::HybridX86ARM64:
  case CPUType::ARM64EC:
  case CPUType::ARM64X:
    return model::Architecture::aarch64;

  default:
    return model::Architecture::Invalid;
  }
}

void PDBImporterImpl::detectArchitecture() {
  revng_log(Log, "detectArchitecture");
  LoggerIndent Indent(Log);

  // Identify the architecture from Compile{2,3}Sym
  model::Architecture::Values Architecture = model::Architecture::Invalid;

  auto ProcessMachine = [&Architecture](CPUType Machine) {
    auto NewArchitecture = toModelArchitecture(Machine);
    if (Architecture == model::Architecture::Invalid) {
      Architecture = NewArchitecture;
    } else if (Architecture != NewArchitecture) {
      revng_log(Log,
                "Warning: PDB reports multiple architectures: "
                  << model::Architecture::getName(NewArchitecture));
    }
  };

  for (auto &Compile2Symbol : Symbols.Compile2Records)
    ProcessMachine(Compile2Symbol.Machine);

  for (auto &Compile3Symbol : Symbols.Compile3Records)
    ProcessMachine(Compile3Symbol.Machine);

  if (Architecture == model::Architecture::Invalid)
    return;

  auto &Model = Importer.getModel();
  if (Model->Architecture() == model::Architecture::Invalid) {
    Model->Architecture() = Architecture;
    if (Model->DefaultABI() == model::ABI::Invalid) {
      if (auto MaybeABI = model::ABI::getDefaultForPECOFF(Architecture)) {
        Model->DefaultABI() = *MaybeABI;
      }
    }
  } else if (Model->Architecture() != Architecture) {
    revng_log(Log,
              "Warning: PDB reports an unexpected architecture: "
                << model::Architecture::getName(Architecture));
  }
}

void PDBImporterImpl::preregisterTypeDefinitions() {
  using namespace model;

  revng_log(Log, "Running preregisterTypeDefinitions");
  LoggerIndent Indent(Log);

  for (auto &[TypeIndex, Object] : TypeRecords.ClassRecords)
    if (not ProcessedTypes.hasBeenProcessed(TypeIndex))
      preregisterTypeDefinition<StructDefinition>(TypeIndex);

  for (auto &[TypeIndex, Object] : TypeRecords.UnionRecords)
    if (not ProcessedTypes.hasBeenProcessed(TypeIndex))
      preregisterTypeDefinition<UnionDefinition>(TypeIndex);

  for (auto &[TypeIndex, Object] : TypeRecords.EnumRecords)
    if (not ProcessedTypes.hasBeenProcessed(TypeIndex))
      preregisterTypeDefinition<EnumDefinition>(TypeIndex);

  for (auto &[TypeIndex, Object] : TypeRecords.ProcedureRecords)
    if (not ProcessedTypes.hasBeenProcessed(TypeIndex))
      preregisterTypeDefinition<CABIFunctionDefinition>(TypeIndex);

  for (auto &[TypeIndex, Object] : TypeRecords.MemberFunctionRecords)
    if (not ProcessedTypes.hasBeenProcessed(TypeIndex))
      preregisterTypeDefinition<CABIFunctionDefinition>(TypeIndex);

  for (auto &[TypeIndex, Object] : TypeRecords.AliasRecords)
    if (not ProcessedTypes.hasBeenProcessed(TypeIndex))
      preregisterTypeDefinition<TypedefDefinition>(TypeIndex);
}

void PDBImporterImpl::populateTypes() {
  using namespace model;

  revng_log(Log, "Running populateTypes");
  LoggerIndent Indent(Log);

  TypeResolver Resolver(*this, Importer.getModel()->Architecture());

  for (auto &[TypeIndex, Object] : TypeRecords.ClassRecords)
    Resolver.getTypeFor(TypeIndex);

  for (auto &[TypeIndex, Object] : TypeRecords.UnionRecords)
    Resolver.getTypeFor(TypeIndex);

  for (auto &[TypeIndex, Object] : TypeRecords.EnumRecords)
    Resolver.getTypeFor(TypeIndex);

  for (auto &[TypeIndex, Object] : TypeRecords.ProcedureRecords)
    Resolver.getTypeFor(TypeIndex);

  for (auto &[TypeIndex, Object] : TypeRecords.MemberFunctionRecords)
    Resolver.getTypeFor(TypeIndex);

  for (auto &[TypeIndex, Object] : TypeRecords.AliasRecords)
    Resolver.getTypeFor(TypeIndex);
}

void PDBImporterImpl::visitSymbols(SymbolVisitorCallbacks &Visitor) {
  auto &PDBFile = *Importer.getPDBFile();

  auto MaybePDIStream = PDBFile.getPDBDbiStream();
  if (auto Error = MaybePDIStream.takeError()) {
    revng_log(Log, "Warning: error obtaining the PDI stream: " << Error);
    consumeError(std::move(Error));
    return;
  }

  uint32_t ModulesCount = MaybePDIStream->modules().getModuleCount();

  for (uint32_t StreamIndex = 0; StreamIndex < ModulesCount; ++StreamIndex) {
    revng_log(Log, "Handling stream " << StreamIndex);
    LoggerIndent Indent(Log);

    auto MaybeDebugStream = getModuleDebugStream(PDBFile, StreamIndex);
    if (llvm::Error Error = MaybeDebugStream.takeError()) {
      std::string Message = llvm::toString(std::move(Error));
      revng_log(Log, "Warning: error reading from the PDB stream: " << Message);
      continue;
    }

    ModuleDebugStreamRef &DebugStream = *MaybeDebugStream;

    SymbolVisitorCallbackPipeline Pipeline;
    SymbolDeserializer Deserializer(nullptr, CodeViewContainer::Pdb);

    // This is not a demangler, this parses the raw bytes
    Pipeline.addCallbackToPipeline(Deserializer);
    Pipeline.addCallbackToPipeline(Visitor);
    CVSymbolVisitor Visitor(Pipeline);
    auto SS = DebugStream.getSymbolsSubstream();
    if (auto Error = Visitor.visitSymbolStream(DebugStream.getSymbolArray(),
                                               SS.Offset)) {
      std::string Message = llvm::toString(std::move(Error));
      revng_log(Log, "Warning: visiting the symbol stream: " << Message);
    }
  }
}

class NameMap {
public:
  using TypeIndex = llvm::codeview::TypeIndex;

public:
  enum class Status {
    NoMatches,
    OneMatch,
    MultipleMatches
  };

private:
  llvm::StringMap<TypeIndex> Definitions;

public:
  void registerName(llvm::StringRef Name, TypeIndex Index) {
    revng_assert(not Index.isNoneType());
    auto It = Definitions.find(Name);
    if (It != Definitions.end()) {
      It->second = TypeIndex::None();
      revng_log(Log, "Warning: multiple definitions of " << Name << " found");
    } else {
      Definitions[Name] = Index;
    }
  }

public:
  std::pair<Status, TypeIndex> get(llvm::StringRef Name) const {
    auto It = Definitions.find(Name);
    if (It == Definitions.end())
      return { Status::NoMatches, TypeIndex::None() };
    else if (It->second.isNoneType())
      return { Status::MultipleMatches, TypeIndex::None() };
    else
      return { Status::OneMatch, It->second };
  }
};

const model::UpcastableType &
PDBImporterImpl::recordType(TypeIndex Index,
                            model::UpcastableType &&Result,
                            model::TypeDefinition *Definition,
                            bool IsSizeAvailable) {
  using namespace llvm;
  using namespace model;

  auto It = Typedefs.find(Index);
  bool IsTypedefd = It != Typedefs.end() and It->second != nullptr;

  // Ensure that if we already processed this type, we processed exactly the
  // same type
  if (const auto *Entry = ProcessedTypes.tryGetEntry(Index)) {
    if (Result == Entry->Type) {
      revng_log(Log,
                toString(Index) << " was already resolved with an identical "
                                   "type");
    } else if (IsTypedefd) {
      auto *Defined = cast<DefinedType>(Entry->Type.get());
      auto *Typedef = cast<TypedefDefinition>(Defined->Definition().get());
      revng_assert(Typedef->UnderlyingType() == Result,
                   (toString(Index)
                    + " was already resolved with a different type")
                     .c_str());
    } else {
      revng_abort((toString(Index)
                   + " was already resolved with a different type")
                    .c_str());
    }

    // Upgrade SizeAvailable if the caller has now ensured children are
    // size-available.
    if (IsSizeAvailable and not Entry->IsSizeAvailable)
      ProcessedTypes.markSizeAvailable(Index);

    return Entry->Type;
  }

  if (IsTypedefd) {
    revng_log(Log, "Using typedef " << It->second->Name.str());
    // This type is associated to one and only one typedef.
    // Emit the typedef on the fly.
    auto [UDTTypedef, Type] = Model->makeTypeDefinition<TypedefDefinition>();
    UDTTypedef.Name() = It->second->Name;
    UDTTypedef.UnderlyingType() = std::move(Result);
    Result = std::move(Type);
    // The Definition pointer (if provided) still points to the inner
    // TypeDefinition to be populated by processDefinition; size-availability
    // of the outer wrapper follows from the inner.
  }

  return ProcessedTypes.record(Index,
                               std::move(Result),
                               Definition,
                               IsSizeAvailable)
    .Type;
}

template<typename T>
void PDBImporterImpl::resolveForwardReferences(StringRef Name,
                                               Records::TypeList<T>
                                                 &ObjectList) {
  revng_log(Log, "Resolving forward declarations for " << Name.str());
  LoggerIndent Indent(Log);

  NameMap UniqueNamesMap;
  NameMap RegularNamesMap;

  // Collect all the full definitions
  for (auto &[Index, Object] : ObjectList) {
    if (Object.isForwardRef())
      continue;

    if (Object.hasUniqueName()) {
      UniqueNamesMap.registerName(Object.getUniqueName(), Index);
    } else {
      RegularNamesMap.registerName(Object.getName(), Index);
    }
  }

  for (auto &[Index, Object] : ObjectList) {
    // Collect all the full definitions
    if (not Object.isForwardRef())
      continue;

    NameMap::Status Status;
    TypeIndex Match;
    StringRef Name;
    if (Object.hasUniqueName()) {
      Name = Object.getUniqueName();
      std::tie(Status, Match) = UniqueNamesMap.get(Name);
    } else {
      Name = Object.getName();
      std::tie(Status, Match) = RegularNamesMap.get(Name);
    }

    revng_log(Log, "Considering " << toString(Index) << ", " << Name.str());
    LoggerIndent Indent(Log);

    if (Status == NameMap::Status::NoMatches) {
      revng_log(Log, "Unmatched forward declaration, typedef'ing to void");
      // TODO: maybe we could create an empty enum here, in case of a enum
      auto [Definition,
            Type] = registerTypeDefinition<model::TypedefDefinition>(Index);
      Definition.Name() = Name;
      Definition.UnderlyingType() = model::PrimitiveType::makeVoid();
      // Definition is now fully populated (UnderlyingType set to void).
      markSizeAvailable(Index);
    } else if (Status == NameMap::Status::MultipleMatches) {
      revng_log(Log, "Warning: ambiguous forward declaration, ignoring");
      recordType(Index, model::UpcastableType::empty(), true);
    } else {
      // Redirect to full definition
      TypeRecords.RecordsByIndex[Index] = TypeRecords.RecordsByIndex[Match];
    }
  }
}

bool PDBImporterImpl::loadInputFile() {
  revng_log(Log, "Running loadInputFile");
  LoggerIndent Indent(Log);

  auto Path = Importer.getPDBFile()->getFilePath();
  auto MaybeInputFile = llvm::pdb::InputFile::open(Path);
  if (not MaybeInputFile) {
    revng_log(Log,
              "Warning: unable to open PDB file "
                << MaybeInputFile.takeError());
    consumeError(MaybeInputFile.takeError());
    return false;
  }

  TheInput = &*MaybeInputFile;
  return true;
}

void PDBImporterImpl::createTypedefs() {
  revng_log(Log, "Running createTypedefs");
  LoggerIndent Indent(Log);
  for (llvm::codeview::UDTSym &UDTSymbol : Symbols.UDTSymRecords) {
    auto It = Typedefs.find(UDTSymbol.Type);
    if (It == Typedefs.end()) {
      Typedefs[UDTSymbol.Type] = &UDTSymbol;
    } else {
      Typedefs[UDTSymbol.Type] = nullptr;
    }
  }
}

void PDBImporterImpl::resolveAllForwardReferences() {
  revng_log(Log, "Running resolveAllForwardReferences");
  LoggerIndent Indent(Log);
  resolveForwardReferences("ClassRecords", TypeRecords.ClassRecords);
  resolveForwardReferences("UnionRecords", TypeRecords.UnionRecords);
  resolveForwardReferences("EnumRecords", TypeRecords.EnumRecords);
}

void PDBImporterImpl::run() {
  revng_log(Log, "Running PDBImporter");
  LoggerIndent Indent(Log);

  if (not loadInputFile())
    return;

  collectRecords();

  detectArchitecture();

  createTypedefs();

  resolveAllForwardReferences();

  preregisterTypeDefinitions();

  populateTypes();

  // Only preregistered entries (Definition != nullptr) need to be size-
  // available: Modifier/BitField wrappers resolved behind a pointer path
  // legitimately stay IsSizeAvailable=false, but since they transitively
  // depend on preregistered definitions, checking those is sufficient.
  bool HasUnresolved = false;
  for (const auto &[Index, Entry] : ProcessedTypes) {
    if (Entry.Definition != nullptr and not Entry.IsSizeAvailable) {
      if (not HasUnresolved) {
        dbg << "The following TypeDefinitions were not resolved:\n";
        HasUnresolved = true;
      }
      dbg << "  " << toString(Index) << ": ";
      Entry.Type->dump();
    }
  }

  if (HasUnresolved)
    revng_abort();

  populateSymbolsWithTypes();

  revng_log(Log, "dropTypesDependingOnDefinitions");
  dropTypesDependingOnDefinitions(Model, ToDrop);

  revng_log(Log, "flattenPrimitiveTypedefs");
  model::flattenPrimitiveTypedefs(Model);

  revng_log(Log, "deduplicateEquivalentTypes");
  deduplicateEquivalentTypes(Model);

  revng_log(Log, "deduplicateCollidingNames");
  model::deduplicateCollidingNames(Model);

  revng_log(Log, "purgeUnreachableTypes");
  purgeUnreachableTypes(Model);

  revng_log(Log, "purgeInvalidTypes");
  model::purgeInvalidTypes(Model);

  revng_assert(Model->verify(true));
}

void PDBImporterImpl::handleProcedureSymbol(ProcSym &Procedure) {
  TypeIndex FunctionTypeIndex = Procedure.FunctionType;

  revng_log(Log,
            "Importing " << Procedure.Name << " with type "
                         << toString(FunctionTypeIndex));
  LoggerIndent Indent(Log);

  // If it is not in the .idata already, we assume it is a static symbol.
  auto &Model = Importer.getModel();

  LocalSymbolNames.insert(Procedure.Name);

  uint64_t FunctionVirtualAddress = Importer.getNativeSession()
                                      ->getRVAFromSectOffset(Procedure.Segment,
                                                             Procedure
                                                               .CodeOffset);

  if (not Importer.isFunctionAllowed(FunctionVirtualAddress)) {
    revng_log(Log, "Ignoring disallowed function");
    return;
  }

  // Relocate the symbol.
  MetaAddress FunctionAddress = Importer.toPC(Importer.getBaseAddress()
                                              + FunctionVirtualAddress);

  if (not Model->Functions().contains(FunctionAddress)) {
    if (auto *Function = Importer.registerFunctionEntry(FunctionAddress)) {
      revng_log(Log, "New function registered");
      Function->Name() = Procedure.Name;
      if (auto *Type = tryGetType(FunctionTypeIndex))
        Function->Prototype() = Type->copy();
    }
  } else {
    auto It = Model->Functions().find(FunctionAddress);
    if (auto *Type = tryGetType(FunctionTypeIndex)) {
      revng_log(Log, "Updating prototype");
      It->Prototype() = Type->copy();
    }
  }

  // TODO: Handle Imported functions
}
