/// \file PDBImporter.cpp

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <optional>

#include "llvm/DebugInfo/CodeView/CVSymbolVisitor.h"
#include "llvm/DebugInfo/CodeView/CVTypeVisitor.h"
#include "llvm/DebugInfo/CodeView/GUID.h"
#include "llvm/DebugInfo/CodeView/LazyRandomTypeCollection.h"
#include "llvm/DebugInfo/CodeView/SymbolDeserializer.h"
#include "llvm/DebugInfo/CodeView/SymbolRecord.h"
#include "llvm/DebugInfo/CodeView/SymbolVisitorCallbackPipeline.h"
#include "llvm/DebugInfo/CodeView/SymbolVisitorCallbacks.h"
#include "llvm/DebugInfo/CodeView/TypeDumpVisitor.h"
#include "llvm/DebugInfo/CodeView/TypeRecordHelpers.h"
#include "llvm/DebugInfo/PDB/Native/DbiStream.h"
#include "llvm/DebugInfo/PDB/Native/GlobalsStream.h"
#include "llvm/DebugInfo/PDB/Native/InfoStream.h"
#include "llvm/DebugInfo/PDB/Native/InputFile.h"
#include "llvm/DebugInfo/PDB/Native/ModuleDebugStream.h"
#include "llvm/DebugInfo/PDB/Native/NativeSession.h"
#include "llvm/DebugInfo/PDB/Native/PDBFile.h"
#include "llvm/DebugInfo/PDB/Native/SymbolStream.h"
#include "llvm/DebugInfo/PDB/Native/TpiStream.h"
#include "llvm/DebugInfo/PDB/PDB.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/Program.h"

#include "revng/Model/Binary.h"
#include "revng/Model/Importer/Binary/Options.h"
#include "revng/Model/Importer/DebugInfo/PDBImporter.h"
#include "revng/Model/Pass/AllPasses.h"
#include "revng/Model/Processing.h"
#include "revng/Model/TypeDefinition.h"
#include "revng/Support/Assert.h"
#include "revng/Support/CommandLine.h"
#include "revng/Support/Debug.h"
#include "revng/Support/MetaAddress.h"
#include "revng/Support/PathList.h"
#include "revng/Support/ProgramRunner.h"

#include "ImportDebugInfoHelper.h"

using namespace llvm;
using namespace llvm::codeview;
using namespace llvm::object;
using namespace llvm::pdb;

static Logger<> Log("pdb-importer");

// Force using a specific PDB.
static llvm::cl::opt<std::string> UsePDB("use-pdb",
                                         llvm::cl::desc("Path to the PDB."),
                                         llvm::cl::cat(MainCategory));

PDBImporter::PDBImporter(TupleTree<model::Binary> &Model,
                         MetaAddress ImageBase) :
  BinaryImporterHelper(*Model, ImageBase.address(), Log),
  Model(Model),
  ImageBase(ImageBase) {

  // When we import debug info, we assume we already have parsed Segments
  processSegments();
}

namespace {
class PDBImporterImpl {
private:
  PDBImporter &Importer;
  DenseMap<TypeIndex, model::UpcastableType> ProcessedTypes;

public:
  PDBImporterImpl(PDBImporter &Importer) : Importer(Importer) {}
  void run(NativeSession &Session);

private:
  void populateTypes();
  void populateSymbolsWithTypes(NativeSession &Session);
};

using ProcessedTypeMap = DenseMap<TypeIndex, model::UpcastableType>;

/// Visitor for CodeView type streams found in PDB files. It overrides callbacks
/// (from `TypeVisitorCallbacks`) to types of interest for the revng `Model`.
/// During the traversal of the graph from PDB that represents the type system,
/// the `Types:` field of the `Model` is being populated. Since each CodeView
/// type in the PDB has unique `TypeIndex` that will be used when a symbol from
/// PDB symbol stream uses a certain type, we also keep a map of such
/// `TypeIndex` to corresponding type generated within the `Model` (it is done
/// by using `ProcessedTypes`), so it can be used when connecting functions from
/// `Model` with corresponding prototypes.
class PDBImporterTypeVisitor : public TypeVisitorCallbacks {
  TupleTree<model::Binary> &Model;
  LazyRandomTypeCollection &Types;
  ProcessedTypeMap &ProcessedTypes;
  DenseMap<TypeIndex, TypeIndex> &ForwardReferencedTypes;
  TpiStream &Tpi;

  TypeIndex CurrentTypeIndex = TypeIndex::None();
  std::map<TypeIndex, SmallVector<DataMemberRecord, 8>> InProgressMemberTypes;
  std::map<TypeIndex, SmallVector<EnumeratorRecord, 8>>
    InProgressEnumeratorTypes;
  std::map<TypeIndex, ArgListRecord> InProgressArgumentsTypes;

  // Methods of a Class type. It references concrete MemberFunctionRecord.
  std::map<TypeIndex, SmallVector<OneMethodRecord, 8>>
    InProgressFunctionMemberTypes;
  DenseMap<TypeIndex, MemberFunctionRecord>
    InProgressConcreteFunctionMemberTypes;

public:
  PDBImporterTypeVisitor(TupleTree<model::Binary> &M,
                         LazyRandomTypeCollection &Types,
                         ProcessedTypeMap &ProcessedTypes,
                         DenseMap<TypeIndex, TypeIndex> &ForwardReferencedTypes,
                         TpiStream &Tpi) :
    TypeVisitorCallbacks(),
    Model(M),
    Types(Types),
    ProcessedTypes(ProcessedTypes),
    ForwardReferencedTypes(ForwardReferencedTypes),
    Tpi(Tpi) {}

  Error visitTypeBegin(CVType &Record) override;
  Error visitTypeBegin(CVType &Record, TypeIndex TI) override;

  Error visitKnownRecord(CVType &Record, ClassRecord &Class) override;
  Error visitKnownMember(CVMemberRecord &Record,
                         EnumeratorRecord &Member) override;
  Error visitKnownRecord(CVType &Record, EnumRecord &Enum) override;
  Error visitKnownRecord(CVType &Record, ProcedureRecord &Proc) override;
  Error visitKnownRecord(CVType &Record, UnionRecord &Union) override;
  Error visitKnownRecord(CVType &Record, ArgListRecord &Args) override;
  Error visitKnownMember(CVMemberRecord &Record,
                         DataMemberRecord &Member) override;
  Error visitKnownRecord(CVType &Record, FieldListRecord &FieldList) override;
  Error visitKnownRecord(CVType &Record, PointerRecord &Ptr) override;
  Error visitKnownRecord(CVType &Record, ModifierRecord &Modifier) override;
  Error visitKnownRecord(CVType &Record, ArrayRecord &Array) override;

  Error visitKnownMember(CVMemberRecord &Record,
                         OneMethodRecord &FnMember) override;
  Error visitKnownRecord(CVType &CVR,
                         MemberFunctionRecord &MemberFnRecord) override;

  model::UpcastableType makeModelTypeForIndex(TypeIndex Index);
  model::UpcastableType createPrimitiveType(TypeIndex SimpleType);
};

/// Visitor for CodeView symbol streams found in PDB files. It is being used for
/// connecting functions from `Model` to their prototypes. We assume the PDB
/// type stream was traversed before invoking this class.
class PDBImporterSymbolVisitor : public SymbolVisitorCallbacks {
private:
  BinaryImporterHelper &Helper;
  TupleTree<model::Binary> &Model;
  ProcessedTypeMap &ProcessedTypes;

  NativeSession &Session;
  MetaAddress &ImageBase;

public:
  PDBImporterSymbolVisitor(BinaryImporterHelper &Helper,
                           TupleTree<model::Binary> &M,
                           ProcessedTypeMap &ProcessedTypes,
                           NativeSession &Session,
                           MetaAddress &ImageBase) :
    Helper(Helper),
    Model(M),
    ProcessedTypes(ProcessedTypes),
    Session(Session),
    ImageBase(ImageBase) {}

  Error visitSymbolBegin(CVSymbol &Record) override;
  Error visitSymbolBegin(CVSymbol &Record, uint32_t Offset) override;
  Error visitKnownRecord(CVSymbol &Record, ProcSym &Proc) override;
};
} // namespace

void PDBImporterImpl::populateTypes() {
  auto MaybeInputFile = InputFile::open(Importer.getPDBFile()->getFilePath());
  if (not MaybeInputFile) {
    revng_log(Log, "Unable to open PDB file " << MaybeInputFile.takeError());
    consumeError(MaybeInputFile.takeError());
    return;
  }

  auto MaybeTpiStream = Importer.getPDBFile()->getPDBTpiStream();
  if (not MaybeTpiStream) {
    revng_log(Log,
              "Unable to find TPI in PDB file: " << MaybeTpiStream.takeError());
    consumeError(MaybeTpiStream.takeError());
    return;
  }

  // Those will be processed after all the types are visited.
  DenseMap<TypeIndex, TypeIndex> ForwardReferencedTypes;
  PDBImporterTypeVisitor TypeVisitor(Importer.getModel(),
                                     MaybeInputFile->types(),
                                     ProcessedTypes,
                                     ForwardReferencedTypes,
                                     *MaybeTpiStream);
  if (auto Error = visitTypeStream(MaybeInputFile->types(), TypeVisitor)) {
    revng_log(Log, "Error during visiting types: " << Error);
    consumeError(std::move(Error));
  }
}

class PDBSymbolHandler {
private:
  PDBImporter &Importer;
  ProcessedTypeMap &ProcessedTypes;
  NativeSession &Session;
  InputFile &Input;

public:
  PDBSymbolHandler(PDBImporter &Importer,
                   ProcessedTypeMap &ProcessedTypes,
                   NativeSession &Session,
                   InputFile &Input) :
    Importer(Importer),
    ProcessedTypes(ProcessedTypes),
    Session(Session),
    Input(Input) {}

  Error operator()(uint32_t Modi, const SymbolGroup &SG) {
    auto MaybeDebugStream = getModuleDebugStream(*Importer.getPDBFile(), Modi);
    if (MaybeDebugStream) {
      ModuleDebugStreamRef &ModS = *MaybeDebugStream;

      SymbolVisitorCallbackPipeline Pipeline;
      SymbolDeserializer Deserializer(nullptr, CodeViewContainer::Pdb);
      PDBImporterSymbolVisitor SymVisitor(Importer,
                                          Importer.getModel(),
                                          ProcessedTypes,
                                          Session,
                                          Importer.getBaseAddress());

      Pipeline.addCallbackToPipeline(Deserializer);
      Pipeline.addCallbackToPipeline(SymVisitor);
      CVSymbolVisitor Visitor(Pipeline);
      auto SS = ModS.getSymbolsSubstream();
      if (auto Err = Visitor.visitSymbolStream(ModS.getSymbolArray(),
                                               SS.Offset))
        return createStringError(errorToErrorCode(std::move(Err)),
                                 Input.getFilePath());
    } else {
      // If the module stream does not exist, it is not an
      // error condition.
      consumeError(MaybeDebugStream.takeError());
    }

    return Error::success();
  }
};

void PDBImporterImpl::populateSymbolsWithTypes(NativeSession &Session) {
  auto MaybeInputFile = InputFile::open(Importer.getPDBFile()->getFilePath());
  if (not MaybeInputFile) {
    revng_log(Log, "Unable to open PDB file: " << MaybeInputFile.takeError());
    consumeError(MaybeInputFile.takeError());
    return;
  }

  FilterOptions Filters{};
  LinePrinter Printer(/*Indent=*/2, false, nulls(), Filters);
  const PrintScope HeaderScope(Printer, /*IndentLevel=*/2);
  PDBSymbolHandler SymbolHandler(Importer,
                                 ProcessedTypes,
                                 Session,
                                 *MaybeInputFile);
  if (auto Error = iterateSymbolGroups(*MaybeInputFile,
                                       HeaderScope,
                                       SymbolHandler)) {
    revng_log(Log, "Unable to parse symbols: " << Error);
    consumeError(std::move(Error));
    return;
  }
}

void PDBImporterImpl::run(NativeSession &Session) {
  populateTypes();
  populateSymbolsWithTypes(Session);

  TupleTree<model::Binary> &Model = Importer.getModel();
  deduplicateEquivalentTypes(Model);
  purgeUnreachableTypes(Model);
  revng_assert(Model->verify(true));
}

bool PDBImporter::loadDataFromPDB(StringRef PDBFileName) {
  auto Err = loadDataForPDB(PDB_ReaderType::Native, PDBFileName, Session);
  if (Err) {
    revng_log(Log, "Unable to read PDB file: " << Err);
    consumeError(std::move(Err));
    return false;
  }

  TheNativeSession = static_cast<NativeSession *>(Session.get());
  // TODO: We are using the static_cast due to lack of an LLVM RTTI
  // support for this. Once it is improved in LLVM, we should avoid this.
  auto SessionLoadAddress = Session->getLoadAddress();
  auto NativeSessionLoadAddress = TheNativeSession->getLoadAddress();
  revng_assert(SessionLoadAddress == NativeSessionLoadAddress);

  ThePDBFile = &TheNativeSession->getPDBFile();

  if (ExpectedGUID) {
    auto MaybePDBInfoStream = ThePDBFile->getPDBInfoStream();
    if (auto Error = MaybePDBInfoStream.takeError()) {
      consumeError(std::move(Error));
      // TODO: is it correct to ignore this error?
      return true;
    }

    codeview::GUID GUIDFromPDBFile = MaybePDBInfoStream->getGuid();
    if (ExpectedGUID != GUIDFromPDBFile) {
      revng_log(Log, "Signatures from exe and PDB file mismatch");
      return false;
    }
  }

  return true;
}

static bool fileExists(const Twine &Path) {
  bool Result = sys::fs::exists(Path);

  if (Result) {
    revng_log(Log, "Found: " << Path.str());
  } else {
    revng_log(Log, "The following path does not exist: " << Path.str());
  }

  return Result;
}

std::optional<std::string>
PDBImporter::getCachedPDBFilePath(std::string PDBFileID,
                                  StringRef PDBBaseName) {
  std::string CacheDir = getCacheDirectory();
  std::string ResultPath = joinPath(CacheDir,
                                    "debug-symbols",
                                    "pe",
                                    PDBFileID,
                                    PDBBaseName);
  if (fileExists(ResultPath))
    return ResultPath;

  return std::nullopt;
}

// Construct PDB file ID.
static std::string formatPDBFileID(ArrayRef<uint8_t> Bytes, uint16_t Age) {
  std::string PDBGUID;
  raw_string_ostream StringPDBGUID(PDBGUID);
  StringPDBGUID << format_bytes(Bytes,
                                /*FirstByteOffset*/ {},
                                /*NumPerLine*/ 16,
                                /*ByteGroupSize*/ 16);
  StringPDBGUID.flush();

  // Let's format the PDB file ID.
  // The PDB GUID is `7209ac2725e5fe841a88b1fe70d1603b` and `Age` is 2.
  // The PDB ID `Hash` is: `27ac0972e52584fe1a88b1fe70d1603b2`.
  std::string PDBFileID;
  PDBFileID += PDBGUID[6];
  PDBFileID += PDBGUID[7];
  PDBFileID += PDBGUID[4];
  PDBFileID += PDBGUID[5];
  PDBFileID += PDBGUID[2];
  PDBFileID += PDBGUID[3];
  PDBFileID += PDBGUID[0];
  PDBFileID += PDBGUID[1];

  PDBFileID += PDBGUID[10];
  PDBFileID += PDBGUID[11];
  PDBFileID += PDBGUID[8];
  PDBFileID += PDBGUID[9];
  PDBFileID += PDBGUID[14];
  PDBFileID += PDBGUID[15];
  PDBFileID += PDBGUID[12];
  PDBFileID += PDBGUID[13];

  PDBFileID += PDBGUID.substr(16);
  PDBFileID += ('0' + Age);

  return PDBFileID;
}

void PDBImporter::import(const COFFObjectFile &TheBinary,
                         const ImporterOptions &Options) {
  if (Options.DebugInfo == DebugInfoLevel::No)
    return;

  auto MaybePDBPath = getPDBFilePath(TheBinary);
  if (not MaybePDBPath)
    return;

  if (not loadDataFromPDB(*MaybePDBPath))
    return;

  PDBImporterImpl ModelCreator(*this);
  ModelCreator.run(*TheNativeSession);
}

static StringRef getBaseName(StringRef Path) {
  auto PositionOfLastDirectoryChar = Path.rfind("\\");
  if (PositionOfLastDirectoryChar != llvm::StringRef::npos) {
    return Path.slice(PositionOfLastDirectoryChar + 1, Path.size());
  }
  return Path;
}

std::optional<std::string>
PDBImporter::getPDBFilePath(const COFFObjectFile &TheBinary) {
  // Consider the --use-pdb argument
  if (not UsePDB.empty()) {
    if (not fileExists(UsePDB)) {
      revng_log(Log, "Argument --use-pdb does not exist, ignoring.");
    } else {
      return UsePDB;
    }
  }

  // Parse debug info in TheBinary
  const codeview::DebugInfo *DebugInfo = nullptr;
  std::string InternalPDBPath;
  {
    StringRef InternalPDBStringReference;
    auto EC = TheBinary.getDebugPDBInfo(DebugInfo, InternalPDBStringReference);
    if (EC) {
      revng_log(Log, "getDebugPDBInfo failed: " << EC);
      consumeError(std::move(EC));
      return std::nullopt;
    } else if (DebugInfo == nullptr) {
      revng_log(Log, "Couldn't get codeview::DebugInfo");
      return std::nullopt;
    } else {
      InternalPDBPath = InternalPDBStringReference.str();
    }
  }

  // TODO: Handle PDB signature types other then PDB70, e.g. PDB20.
  if (DebugInfo->Signature.CVSignature != OMF::Signature::PDB70) {
    revng_log(Log, "A non-PDB70 signature was find, ignore.");
    return std::nullopt;
  }

  // According to llvm/docs/PDB/PdbStream.rst, the `Signature` was never
  // used the way as it was the initial idea. Instead, GUID is a 128-bit
  // identifier guaranteed to be unique ID for both executable and
  // corresponding PDB. Save the GUID for later to check the match.
  ExpectedGUID = llvm::codeview::GUID();
  llvm::copy(DebugInfo->PDB70.Signature, std::begin(ExpectedGUID->Guid));

  if (InternalPDBPath.empty()) {
    revng_log(Log, "The internal PDB path is empty");
    return std::nullopt;
  }

  // If the internal PDB path exists, use that
  if (fileExists(InternalPDBPath)) {
    return InternalPDBPath;
  }

  // The path specified in the binary does not exist: extract the file name and
  // look for it in other (canonical) places

  StringRef PDBBaseName = getBaseName(InternalPDBPath);

  // Try in the current directory
  llvm::SmallString<128> ResultPath;
  if (auto ErrorCode = llvm::sys::fs::current_path(ResultPath)) {
    revng_log(Log, "Can't get current working path.");
  } else {
    llvm::sys::path::append(ResultPath, PDBBaseName);
    if (fileExists(ResultPath.str()))
      return ResultPath.str().str();
  }

  // Try main input path
  ResultPath.clear();
  if (not InputPath.empty()) {
    llvm::sys::path::append(ResultPath,
                            llvm::sys::path::parent_path(InputPath),
                            PDBBaseName);
    if (fileExists(ResultPath.str()))
      return ResultPath.str().str();
  }

  // Compute the PDB file ID
  auto PDBFileID = formatPDBFileID(DebugInfo->PDB70.Signature,
                                   DebugInfo->PDB70.Age);

  // Check if we already fetched it from PDB servers in the past
  if (auto MaybeCachedPDBPath = getCachedPDBFilePath(PDBFileID, PDBBaseName))
    return MaybeCachedPDBPath;

  // Let's try finding it on web with the `fetch-debuginfo` tool.
  // If the `revng` cannot be found, avoid finding debug info.
  int ExitCode = runFetchDebugInfo(TheBinary.getFileName(), Log.isEnabled());
  if (ExitCode != 0) {
    revng_log(Log,
              "Failed to find debug info with `revng model "
              "fetch-debuginfo`.");
    return std::nullopt;
  }

  // Try again to find the file
  return getCachedPDBFilePath(PDBFileID, PDBBaseName);
}

// ==== Implementation of the Model type recordings. ==== //

Error PDBImporterTypeVisitor::visitTypeBegin(CVType &Record) {
  return visitTypeBegin(Record, TypeIndex::fromArrayIndex(Types.size()));
}

Error PDBImporterTypeVisitor::visitTypeBegin(CVType &Record, TypeIndex TI) {
  CurrentTypeIndex = TI;
  return Error::success();
}

Error PDBImporterTypeVisitor::visitKnownRecord(CVType &Record,
                                               FieldListRecord &FieldList) {
  if (auto EC = visitMemberRecordStream(FieldList.Data, *this))
    return EC;
  return Error::success();
}

// Determine the pointer size based on CodeView/PDB data.
static uint32_t getPointerSize(codeview::PointerKind K) {
  switch (K) {
  case codeview::PointerKind::Near64:
    return 8;
  case codeview::PointerKind::Near32:
    return 4;
  default:
    // TODO: Handle all pointer kinds.
    revng_abort();
  }
}

// Parse LF_POINTER.
Error PDBImporterTypeVisitor::visitKnownRecord(CVType &Record,
                                               PointerRecord &Ptr) {
  TypeIndex ReferencedType = Ptr.getReferentType();
  auto ReferencedTypeFromModel = makeModelTypeForIndex(ReferencedType);
  if (ReferencedTypeFromModel.isEmpty()) {
    revng_log(Log,
              "LF_POINTER: Unknown referenced type "
                << ReferencedType.getIndex());
    return Error::success();
  }

  auto Pointer = model::PointerType::make(std::move(ReferencedTypeFromModel),
                                          getPointerSize(Ptr.getPointerKind()));

  auto &&[Typedef, NewType] = Model->makeTypedefDefinition();
  Typedef.UnderlyingType() = std::move(Pointer);
  ProcessedTypes[CurrentTypeIndex] = std::move(NewType);

  return Error::success();
}

// Parse LF_ARRAY.
Error PDBImporterTypeVisitor::visitKnownRecord(CVType &Record,
                                               ArrayRecord &Array) {
  TypeIndex ElementType = Array.getElementType();
  auto ElementTypeFromModel = makeModelTypeForIndex(ElementType);
  if (ElementTypeFromModel.isEmpty()) {
    revng_log(Log, "LF_ARRAY: Unknown element type " << ElementType.getIndex());
  } else {
    auto MaybeSize = ElementTypeFromModel->size();
    if (not MaybeSize or *MaybeSize == 0 or Array.getSize() == 0) {
      revng_log(Log, "Skipping 0-sized array.");
      return Error::success();
    }

    const uint64_t ArraySize = Array.getSize() / *MaybeSize;
    auto NewA = model::ArrayType::make(std::move(ElementTypeFromModel),
                                       ArraySize);
    auto &&[_, NewType] = Model->makeTypedefDefinition(std::move(NewA));
    ProcessedTypes[CurrentTypeIndex] = std::move(NewType);
  }

  return Error::success();
}

// Parse LF_MODIFIER.
Error PDBImporterTypeVisitor::visitKnownRecord(CVType &Record,
                                               ModifierRecord &Modifier) {
  TypeIndex ReferencedType = Modifier.getModifiedType();

  auto ModelType = makeModelTypeForIndex(ReferencedType);
  if (ModelType.isEmpty()) {
    revng_log(Log,
              "LF_MODIFIER: Unknown referenced type "
                << ReferencedType.getIndex());
  } else {
    using ModifierOs = ModifierOptions;
    if ((Modifier.getModifiers() & ModifierOs::Const) != ModifierOs::None) {
      auto &&[_, NewType] = Model->makeTypedefDefinition(std::move(ModelType));
      NewType->IsConst() = true;
      ProcessedTypes[CurrentTypeIndex] = std::move(NewType);
    }
  }

  return Error::success();
}

// Parse LF_MEMBER.
Error PDBImporterTypeVisitor::visitKnownMember(CVMemberRecord &Record,
                                               DataMemberRecord &Member) {
  InProgressMemberTypes[CurrentTypeIndex].push_back(Member);

  return Error::success();
}

llvm::Error
PDBImporterTypeVisitor::visitKnownRecord(CVType &CVR,
                                         MemberFunctionRecord &MemberFnRecord) {
  InProgressConcreteFunctionMemberTypes[CurrentTypeIndex] = MemberFnRecord;
  return Error::success();
}

// Parse LF_ONEMETHOD.
// This occurs within LF_CLASS and it references an LF_MFUNCTION.
Error PDBImporterTypeVisitor::visitKnownMember(CVMemberRecord &Record,
                                               OneMethodRecord &FnMember) {
  InProgressFunctionMemberTypes[CurrentTypeIndex].push_back(FnMember);
  return Error::success();
}

// Parse LF_ENUMERATE.
Error PDBImporterTypeVisitor::visitKnownMember(CVMemberRecord &Record,
                                               EnumeratorRecord &Member) {
  InProgressEnumeratorTypes[CurrentTypeIndex].push_back(Member);
  return Error::success();
}

// LF_CLASS, LF_STRUCTURE, LF_INTERFACE (TPI)
Error PDBImporterTypeVisitor::visitKnownRecord(CVType &Record,
                                               ClassRecord &Class) {
  using namespace model;

  if (isUdtForwardRef(Record)) {
    Expected<TypeIndex> FD = Tpi.findFullDeclForForwardRef(CurrentTypeIndex);
    if (auto Error = FD.takeError()) {
      revng_log(Log,
                "LF_STRUCTURE: Cannot resolve forward reference for index "
                  << CurrentTypeIndex.getIndex() << ": " << Error);
      consumeError(std::move(Error));
      return Error::success();
    }

    // Remember forward reference, so we can process it later.
    ForwardReferencedTypes[*FD] = CurrentTypeIndex;
    uint64_t ForwardTypeSize = getSizeInBytesForTypeRecord(Tpi.getType(*FD));

    if (ForwardTypeSize == 0) {
      // 0-sized structs are typedef'ed to void. It can happen that there is
      // an incomplete struct type.
      model::UpcastableType Void = model::PrimitiveType::makeVoid();
      auto &&[Typedef, NewType] = Model->makeTypedefDefinition(std::move(Void));
      Typedef.Name() = Class.getName();
      ProcessedTypes[CurrentTypeIndex] = std::move(NewType);
    } else {
      // Pre-create the type that is being referenced by this type.
      auto &&[Struct, NewType] = Model->makeStructDefinition();
      Struct.Name() = Class.getName();
      Struct.Size() = ForwardTypeSize;
      ProcessedTypes[CurrentTypeIndex] = std::move(NewType);
    }

    return Error::success();
  }

  TypeIndex FieldsTypeIndex = Class.getFieldList();
  bool WasReferenced = ForwardReferencedTypes.count(CurrentTypeIndex) != 0;
  if (InProgressMemberTypes.count(FieldsTypeIndex) != 0) {
    model::StructDefinition *Struct = nullptr;
    auto NewDefinition = makeTypeDefinition<model::StructDefinition>();
    if (not WasReferenced) {
      NewDefinition->Name() = Class.getName();
      auto &NewStruct = llvm::cast<model::StructDefinition>(*NewDefinition);
      NewStruct.Size() = Class.getSize();
      Struct = &NewStruct;
    } else {
      TypeIndex ForwardRef = ForwardReferencedTypes[CurrentTypeIndex];
      Struct = &ProcessedTypes[ForwardRef]->toStruct();
    }

    auto &TheFields = InProgressMemberTypes[FieldsTypeIndex];
    uint64_t MaxOffset = 0;

    for (const auto &Field : TheFields) {
      // Create new field.
      uint64_t Offset = Field.getFieldOffset();
      auto FieldModelType = makeModelTypeForIndex(Field.getType());
      if (FieldModelType.isEmpty()) {
        revng_log(Log,
                  "LF_STRUCTURE: Unknown field type "
                    << Field.getType().getIndex());
        // Avoid incomplete struct types.
        return Error::success();
      } else {
        auto MaybeSize = FieldModelType->size();
        uint64_t Size = MaybeSize.value_or(0);
        if (Size == 0) {
          // Skip 0-sized field.
          revng_log(Log, "Skipping 0-sized struct field.");
          continue;
        }

        // This is weird, but I've faced something like:
        // PDB struct TYPE {
        //    offset_0: "sign" // size 1
        //    offset_1: "Local" // size 1
        //
        //    // and again
        //    offset_0: "signLocal" // size 2
        // }
        uint64_t CurrFieldOffset = Offset + Size;
        if (CurrFieldOffset > MaxOffset)
          MaxOffset = CurrFieldOffset;
        else
          continue;

        // TODO: How is this possible?
        // Triggers:
        // `Last field ends outside the struct`.
        if (CurrFieldOffset > Struct->Size()) {
          revng_log(Log, "Skipping struct field that is outside the struct.");
          continue;
        }

        auto &FieldType = Struct->Fields()[Offset];
        FieldType.Name() = Field.getName().str();
        FieldType.Type() = std::move(FieldModelType);
      }
    }

    if (not WasReferenced) {
      auto &&[_, NewType] = Model->recordNewType(std::move(NewDefinition));
      ProcessedTypes[CurrentTypeIndex] = std::move(NewType);
    } else {
      TypeIndex ForwardRef = ForwardReferencedTypes[CurrentTypeIndex];
      ProcessedTypes[CurrentTypeIndex] = ProcessedTypes[ForwardRef].copy();
    }
  }

  // Process methods. Create C-like function prototype for it.
  if (InProgressFunctionMemberTypes.contains(FieldsTypeIndex)) {
    auto &TheFunctions = InProgressFunctionMemberTypes[FieldsTypeIndex];
    for (auto &Function : TheFunctions) {
      TypeIndex FnTypeIndex = Function.getType();
      if (InProgressConcreteFunctionMemberTypes.count(FnTypeIndex) == 0)
        continue;

      // Get the proper LF_MFUNCTION.
      auto &MemberFunction = InProgressConcreteFunctionMemberTypes[FnTypeIndex];
      TypeIndex ReturnTypeIndex = MemberFunction.ReturnType;
      auto ModelReturnType = makeModelTypeForIndex(ReturnTypeIndex);
      if (ModelReturnType.isEmpty()) {
        revng_log(Log,
                  "LF_MFUNCTION: Unknown return type "
                    << ReturnTypeIndex.getIndex());
        // Avoid function types that have incomplete type.
        return Error::success();
      }

      auto NewDefinition = makeTypeDefinition<CABIFunctionDefinition>();
      auto &Prototype = *cast<CABIFunctionDefinition>(NewDefinition.get());
      Prototype.ABI() = Model->DefaultABI();

      if (!ModelReturnType.isEmpty() && !ModelReturnType->isVoidPrimitive())
        Prototype.ReturnType() = std::move(ModelReturnType);

      TypeIndex ArgListTyIndex = MemberFunction.getArgumentList();
      revng_assert(InProgressArgumentsTypes.contains(ArgListTyIndex));
      auto ArgList = InProgressArgumentsTypes[ArgListTyIndex];

      auto Indices = ArgList.getIndices();
      uint32_t Size = Indices.size();

      // Add `this` pointer as an argument if the method is not marked
      // as `static` or `friend`.
      if (Function.getMethodKind() != MethodKind::Static
          and Function.getMethodKind() != MethodKind::Friend
          and ProcessedTypes.count(CurrentTypeIndex) != 0) {
        revng_assert(ProcessedTypes[CurrentTypeIndex].get());
        auto MaybeSize = ProcessedTypes[CurrentTypeIndex].get()->size();
        if (MaybeSize and *MaybeSize != 0) {
          model::UpcastableType T = ProcessedTypes[CurrentTypeIndex].copy();
          auto &Architecture = Model->Architecture();
          Prototype.addArgument(model::PointerType::make(std::move(T),
                                                         Architecture));
        } else {
          revng_log(Log, "Skipping 0-sized argument.");
        }
      }

      for (uint32_t I = 0; I < Size; ++I) {
        TypeIndex ArgumentTypeIndex = Indices[I];
        auto ArgumentTypeFromModel = makeModelTypeForIndex(ArgumentTypeIndex);
        if (not ArgumentTypeFromModel) {
          revng_log(Log,
                    "LF_MFUNCTION: Unknown arg type "
                      << ArgumentTypeIndex.getIndex());
          // Avoid function types that have incomplete type.
          return Error::success();
        } else {
          auto MaybeSize = ArgumentTypeFromModel->size();
          uint64_t Size = MaybeSize.value_or(0);
          if (Size == 0) {
            // Skip 0-sized type.
            revng_log(Log, "Skipping 0-sized argument.");
            continue;
          }

          Prototype.addArgument(std::move(ArgumentTypeFromModel));
        }
      }

      auto &&[_, NewType] = Model->recordNewType(std::move(NewDefinition));
      ProcessedTypes[FnTypeIndex] = std::move(NewType);
    }
  }

  return Error::success();
}

// LF_ENUM (TPI)
Error PDBImporterTypeVisitor::visitKnownRecord(CVType &Record,
                                               EnumRecord &Enum) {
  TypeIndex FieldsTypeIndex = Enum.getFieldList();
  auto NewDefinition = model::makeTypeDefinition<model::EnumDefinition>();
  auto &NewEnum = *cast<model::EnumDefinition>(NewDefinition.get());
  NewEnum.Name() = Enum.getName();

  TypeIndex UnderlyingTypeIndex = Enum.getUnderlyingType();
  auto UnderlyingModelType = makeModelTypeForIndex(UnderlyingTypeIndex);
  if (not UnderlyingModelType) {
    revng_log(Log,
              "LF_ENUM: Unknown underlying type "
                << UnderlyingTypeIndex.getIndex());
    return Error::success();
  }

  NewEnum.UnderlyingType() = std::move(UnderlyingModelType);

  auto &TheFields = InProgressEnumeratorTypes[FieldsTypeIndex];
  if (TheFields.empty())
    return Error::success();

  for (const auto &Entry : TheFields) {
    auto &EnumEntry = NewEnum.Entries()[Entry.getValue().getExtValue()];
    EnumEntry.Name() = Entry.getName().str();
  }

  auto &&[_, NewType] = Model->recordNewType(std::move(NewDefinition));
  ProcessedTypes[CurrentTypeIndex] = std::move(NewType);

  return Error::success();
}

static inline constexpr model::ABI::Values
getMicrosoftABI(CallingConvention CallConv, model::Architecture::Values Arch) {
  if (Arch == model::Architecture::x86_64) {
    switch (CallConv) {
    case CallingConvention::NearC:
    case CallingConvention::NearFast:
    case CallingConvention::NearStdCall:
    case CallingConvention::NearSysCall:
    case CallingConvention::ThisCall:
      return model::ABI::Microsoft_x86_64;
    case CallingConvention::NearPascal:
      revng_abort("Pascal is not currently supported");
    case CallingConvention::NearVector:
      return model::ABI::Microsoft_x86_64_vectorcall;
    case CallingConvention::ClrCall:
      revng_abort("ClrCall is not currently supported");
    default:
      revng_abort();
    }
  } else if (Arch == model::Architecture::x86) {
    switch (CallConv) {
    case CallingConvention::NearC:
      return model::ABI::Microsoft_x86_cdecl;
    case CallingConvention::NearFast:
      return model::ABI::Microsoft_x86_fastcall;
    case CallingConvention::NearStdCall:
      return model::ABI::Microsoft_x86_stdcall;
    case CallingConvention::NearSysCall:
      return model::ABI::Microsoft_x86_stdcall;
    case CallingConvention::ThisCall:
      return model::ABI::Microsoft_x86_thiscall;
    case CallingConvention::ClrCall:
      revng_abort("ClrCall is not currently supported");
    case CallingConvention::NearPascal:
      revng_abort("Pascal is not currently supported");
    case CallingConvention::NearVector:
      return model::ABI::Microsoft_x86_vectorcall;
    default:
      revng_abort();
    }
  } else if (Arch == model::Architecture::mips
             and CallConv == CallingConvention::MipsCall) {
    return model::ABI::SystemV_MIPS_o32;
  } else if (Arch == model::Architecture::mipsel
             and CallConv == CallingConvention::MipsCall) {
    return model::ABI::SystemV_MIPSEL_o32;
  } else if (Arch == model::Architecture::arm
             and CallConv == CallingConvention::ArmCall) {
    return model::ABI::AAPCS;
  } else if (Arch == model::Architecture::aarch64
             /* and CallConv == CallingConvention::ArmCall
                (I'm seeing CallingConvention::NearC)
             */) {
    return model::ABI::Microsoft_AAPCS64;
  } else {
    revng_abort();
  }
}

// LF_PROCEDURE (TPI)
Error PDBImporterTypeVisitor::visitKnownRecord(CVType &Record,
                                               ProcedureRecord &Proc) {
  TypeIndex ReturnTypeIndex = Proc.ReturnType;
  auto ModelReturnType = makeModelTypeForIndex(ReturnTypeIndex);
  if (not ModelReturnType) {
    revng_log(Log,
              "LF_PROCEDURE: Unknown return type "
                << ReturnTypeIndex.getIndex());
  } else {
    auto NewDef = model::makeTypeDefinition<model::CABIFunctionDefinition>();
    auto Prototype = cast<model::CABIFunctionDefinition>(NewDef.get());
    Prototype->ABI() = getMicrosoftABI(Proc.getCallConv(),
                                       Model->Architecture());

    if (!ModelReturnType.isEmpty() && !ModelReturnType->isVoidPrimitive())
      Prototype->ReturnType() = std::move(ModelReturnType);

    TypeIndex ArgListTyIndex = Proc.getArgumentList();
    auto ArgumentList = InProgressArgumentsTypes[ArgListTyIndex];

    auto Indices = ArgumentList.getIndices();
    uint32_t Size = Indices.size();
    for (uint32_t I = 0; I < Size; ++I) {
      TypeIndex ArgumentTypeIndex = Indices[I];
      if (ArgumentTypeIndex.isNoneType()) {
        revng_log(Log,
                  "LF_PROCEDURE: A NoneType argument type "
                    << ArgumentTypeIndex.getIndex());
        continue;
      }

      auto ArgumentTypeFromModel = makeModelTypeForIndex(ArgumentTypeIndex);
      if (not ArgumentTypeFromModel) {
        revng_log(Log,
                  "LF_PROCEDURE: Unknown argument type "
                    << ArgumentTypeIndex.getIndex());
        // Avoid incomplete function types.
        return Error::success();
      } else {
        auto MaybeSize = ArgumentTypeFromModel->size();
        uint64_t Size = MaybeSize.value_or(0);
        // Forward references are processed later.
        if (Size == 0 and !isUdtForwardRef(Tpi.getType(ArgumentTypeIndex))) {
          // Skip 0-sized type.
          revng_log(Log, "Skipping 0-sized argument.");
          continue;
        }

        Prototype->addArgument(std::move(ArgumentTypeFromModel));
      }
    }

    auto &&[_, NewType] = Model->recordNewType(std::move(NewDef));
    ProcessedTypes[CurrentTypeIndex] = std::move(NewType);
  }

  return Error::success();
}

// LF_UNION (TPI)
Error PDBImporterTypeVisitor::visitKnownRecord(CVType &Record,
                                               UnionRecord &Union) {
  TypeIndex FieldsTypeIndex = Union.getFieldList();
  auto &TheFields = InProgressMemberTypes[FieldsTypeIndex];
  if (TheFields.size() == 0) {
    // Handle an empty union, similar to 0-sized structs.
    // Typedef it to void.
    model::UpcastableType Void = model::PrimitiveType::makeVoid();
    auto &&[Typedef, NewType] = Model->makeTypedefDefinition(std::move(Void));
    Typedef.Name() = Union.getName().str();
    ProcessedTypes[CurrentTypeIndex] = std::move(NewType);

    return Error::success();
  }

  auto NewDefinition = model::makeTypeDefinition<model::UnionDefinition>();
  auto &NewUnion = llvm::cast<model::UnionDefinition>(*NewDefinition.get());
  NewUnion.Name() = Union.getName().str();

  bool GeneratedAtLeastOneField = false;
  for (const auto &Field : TheFields) {
    // Create new field.
    auto FieldModelType = makeModelTypeForIndex(Field.getType());
    if (FieldModelType.isEmpty()) {
      revng_log(Log,
                "LF_UNION: Unknown field type " << Field.getType().getIndex());
      // Avoid incomplete unions.
      return Error::success();
    } else {
      uint64_t Size = FieldModelType->size().value_or(0);
      if (Size == 0) {
        // Skip 0-sized field.
        revng_log(Log, "Skipping 0-sized union field.");
        continue;
      }

      GeneratedAtLeastOneField = true;
      auto &FieldType = NewUnion.addField(std::move(FieldModelType));
      FieldType.Name() = Field.getName().str();
    }
  }

  if (GeneratedAtLeastOneField) {
    auto &&[_, NewType] = Model->recordNewType(std::move(NewDefinition));
    ProcessedTypes[CurrentTypeIndex] = std::move(NewType);
  }

  return Error::success();
}

Error PDBImporterTypeVisitor::visitKnownRecord(CVType &Record,
                                               ArgListRecord &Args) {
  InProgressArgumentsTypes[CurrentTypeIndex] = Args;
  return Error::success();
}

// TODO: This can go into LLVM, but there is an ongoing review that should
// implement this.
static std::optional<uint64_t> getSizeinBytes(TypeIndex TI) {
  if (not TI.isSimple())
    return std::nullopt;
  switch (TI.getSimpleKind()) {
  case SimpleTypeKind::Void:
    return 0;
  case SimpleTypeKind::HResult:
    return 4;
  case SimpleTypeKind::SByte:
  case SimpleTypeKind::Byte:
    return 1;
  case SimpleTypeKind::Int16Short:
  case SimpleTypeKind::UInt16Short:
  case SimpleTypeKind::Int16:
  case SimpleTypeKind::UInt16:
    return 2;
  case SimpleTypeKind::Int32Long:
  case SimpleTypeKind::UInt32Long:
  case SimpleTypeKind::Int32:
  case SimpleTypeKind::UInt32:
    return 4;
  case SimpleTypeKind::Int64Quad:
  case SimpleTypeKind::UInt64Quad:
  case SimpleTypeKind::Int64:
  case SimpleTypeKind::UInt64:
    return 8;
  case SimpleTypeKind::Int128Oct:
  case SimpleTypeKind::UInt128Oct:
  case SimpleTypeKind::Int128:
  case SimpleTypeKind::UInt128:
    return 16;
  case SimpleTypeKind::SignedCharacter:
  case SimpleTypeKind::UnsignedCharacter:
  case SimpleTypeKind::NarrowCharacter:
    return 1;
  case SimpleTypeKind::WideCharacter:
  case SimpleTypeKind::Character16:
    return 2;
  case SimpleTypeKind::Character32:
    return 4;
  case SimpleTypeKind::Float16:
    return 2;
  case SimpleTypeKind::Float32:
    return 4;
  case SimpleTypeKind::Float64:
    return 8;
  case SimpleTypeKind::Float80:
    return 10;
  case SimpleTypeKind::Float128:
    return 16;
  case SimpleTypeKind::Boolean8:
    return 1;
  case SimpleTypeKind::Boolean16:
    return 2;
  case SimpleTypeKind::Boolean32:
    return 4;
  case SimpleTypeKind::Boolean64:
    return 8;
  case SimpleTypeKind::Boolean128:
    return 16;
  default:
    return std::nullopt;
  }
}

static model::PrimitiveKind::Values
codeviewSimpleTypeEncodingToModel(TypeIndex TI) {
  if (not TI.isSimple())
    return model::PrimitiveKind::Invalid;

  switch (TI.getSimpleKind()) {
  case SimpleTypeKind::Void:
    return model::PrimitiveKind::Void;
  case SimpleTypeKind::Boolean8:
  case SimpleTypeKind::Boolean16:
  case SimpleTypeKind::Boolean32:
  case SimpleTypeKind::Boolean64:
  case SimpleTypeKind::Boolean128:
  case SimpleTypeKind::Byte:
  case SimpleTypeKind::UInt16:
  case SimpleTypeKind::UInt32:
  case SimpleTypeKind::UInt64:
  case SimpleTypeKind::UnsignedCharacter:
  case SimpleTypeKind::UInt16Short:
  case SimpleTypeKind::UInt32Long:
  case SimpleTypeKind::UInt64Quad:
  case SimpleTypeKind::UInt128Oct:
  case SimpleTypeKind::UInt128:
    return model::PrimitiveKind::Unsigned;
  case SimpleTypeKind::SignedCharacter:
  case SimpleTypeKind::WideCharacter:
  case SimpleTypeKind::NarrowCharacter:
  case SimpleTypeKind::Character16:
  case SimpleTypeKind::Character32:
  case SimpleTypeKind::Int16:
  case SimpleTypeKind::Int16Short:
  case SimpleTypeKind::SByte:
  case SimpleTypeKind::Int32Long:
  case SimpleTypeKind::Int32:
  case SimpleTypeKind::Int64Quad:
  case SimpleTypeKind::Int64:
  case SimpleTypeKind::Int128Oct:
  case SimpleTypeKind::Int128:
    return model::PrimitiveKind::Signed;
  case SimpleTypeKind::Float16:
  case SimpleTypeKind::Float32:
  case SimpleTypeKind::Float64:
  case SimpleTypeKind::Float80:
  case SimpleTypeKind::Float128:
    return model::PrimitiveKind::Float;
  default:
    return model::PrimitiveKind::Invalid;
  }
}

static bool isPointer(TypeIndex TI) {
  if (TI.getSimpleMode() != SimpleTypeMode::Direct) {
    // We have a native pointer.
    switch (TI.getSimpleMode()) {
    case SimpleTypeMode::NearPointer32:
    case SimpleTypeMode::FarPointer32:
    case SimpleTypeMode::NearPointer64:
      return true;
    default:
      return false;
    }
  }

  return false;
}

static bool isTwoBytesLongPointer(TypeIndex TI) {
  if (TI.getSimpleMode() != SimpleTypeMode::Direct) {
    // We have a native pointer.
    switch (TI.getSimpleMode()) {
    case SimpleTypeMode::NearPointer:
    case SimpleTypeMode::FarPointer:
    case SimpleTypeMode::HugePointer:
      return true;
    default:
      return false;
    }
  }
  return false;
}

static bool isSixteenBytesLongPointer(TypeIndex TI) {
  if (TI.getSimpleMode() != SimpleTypeMode::Direct) {
    // We have a native pointer.
    switch (TI.getSimpleMode()) {
    case SimpleTypeMode::NearPointer128:
      return true;
    default:
      return false;
    }
  }
  return false;
}

static std::optional<uint64_t> getPointerSizeFromPDB(TypeIndex TI) {
  if (TI.getSimpleMode() != SimpleTypeMode::Direct) {
    // We have a native pointer.
    switch (TI.getSimpleMode()) {
    case SimpleTypeMode::NearPointer:
    case SimpleTypeMode::FarPointer:
    case SimpleTypeMode::HugePointer:
      return 2;
    case SimpleTypeMode::NearPointer32:
    case SimpleTypeMode::FarPointer32:
      return 4;
    case SimpleTypeMode::NearPointer64:
      return 8;
    case SimpleTypeMode::NearPointer128:
      return 16;
    default:
      return std::nullopt;
    }
  }
  return std::nullopt;
}

model::UpcastableType
PDBImporterTypeVisitor::createPrimitiveType(TypeIndex SimpleType) {
  if (isTwoBytesLongPointer(SimpleType)) {
    // If it is a pointer of size 2, lets create a PointerOrNumber for it.
    using PT = model::PrimitiveType;
    constexpr uint64_t MSDOS16Pointer = 2;
    return ProcessedTypes[SimpleType] = PT::makePointerOrNumber(MSDOS16Pointer);

  } else if (isSixteenBytesLongPointer(SimpleType)) {
    // If it is a 128-bit long pointer, typedef it to void for now. It can be
    // represented as a `struct { pointee; offset; }` since it is how it is
    // implemented in the msvc compiler.
    revng_abort("128-bit pointers are not supported for now.");
    model::UpcastableType Void = model::PrimitiveType::makeVoid();
    auto &&[_, Typedef] = Model->makeTypedefDefinition(std::move(Void));
    return ProcessedTypes[SimpleType] = std::move(Typedef);

  } else {
    auto PrimitiveKind = codeviewSimpleTypeEncodingToModel(SimpleType);
    auto PrimitiveSize = getSizeinBytes(SimpleType);
    if (PrimitiveSize and PrimitiveKind != model::PrimitiveKind::Invalid) {
      auto Primitive = model::PrimitiveType::make(PrimitiveKind,
                                                  *PrimitiveSize);

      if (isPointer(SimpleType)) {
        auto PointerSize = getPointerSizeFromPDB(SimpleType);
        if (!PointerSize) {
          revng_log(Log, "Invalid pointer size " << SimpleType.getIndex());
          return model::UpcastableType::empty();
        }

        auto Pointer = model::PointerType::make(std::move(Primitive),
                                                *PointerSize);
        auto Typedef = Model->makeTypedefDefinition(std::move(Pointer)).second;
        return ProcessedTypes[SimpleType] = std::move(Typedef);
      } else {
        // If it is not a pointer `SimpleKind` will be the same as `SimpleType`.
        revng_assert(TypeIndex(SimpleType.getSimpleKind()) == SimpleType);
        return ProcessedTypes[SimpleType] = std::move(Primitive);
      }
    } else {
      revng_log(Log, "Invalid simple type " << SimpleType.getIndex());
      return model::UpcastableType::empty();
    }
  }
}

model::UpcastableType
PDBImporterTypeVisitor::makeModelTypeForIndex(TypeIndex Index) {
  if (Index.isSimple())
    return createPrimitiveType(Index);

  if (auto Iter = ProcessedTypes.find(Index); Iter != ProcessedTypes.end())
    return Iter->second.copy();
  else
    return model::UpcastableType::empty();
}

// ==== Implementation of the Model Symbol-type connection. ==== //

Error PDBImporterSymbolVisitor::visitSymbolBegin(CVSymbol &Record) {
  return visitSymbolBegin(Record, 0);
}

Error PDBImporterSymbolVisitor::visitSymbolBegin(CVSymbol &Record,
                                                 uint32_t Offset) {
  return Error::success();
}

Error PDBImporterSymbolVisitor::visitKnownRecord(CVSymbol &Record,
                                                 ProcSym &Proc) {
  revng_log(Log, "Importing " << Proc.Name);

  // If it is not in the .idata already, we assume it is a static symbol.
  if (not Model->ImportedDynamicFunctions().contains(Proc.Name.str())) {
    uint64_t FunctionVirtualAddress = Session
                                        .getRVAFromSectOffset(Proc.Segment,
                                                              Proc.CodeOffset);
    // Relocate the symbol.
    MetaAddress FunctionAddress = ImageBase + FunctionVirtualAddress;

    if (not Model->Functions().contains(FunctionAddress)) {
      if (auto *Function = Helper.registerFunctionEntry(FunctionAddress)) {
        Function->Name() = Proc.Name;
        TypeIndex FunctionTypeIndex = Proc.FunctionType;
        if (ProcessedTypes.find(FunctionTypeIndex) != ProcessedTypes.end())
          Function->Prototype() = ProcessedTypes[FunctionTypeIndex];
      }
    } else {
      auto It = Model->Functions().find(FunctionAddress);
      TypeIndex FunctionTypeIndex = Proc.FunctionType;
      if (ProcessedTypes.find(FunctionTypeIndex) != ProcessedTypes.end())
        It->Prototype() = ProcessedTypes[FunctionTypeIndex];
    }
  }

  // TODO: Handle Imported functions.

  return Error::success();
}
