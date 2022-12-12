/// \file PDBImporter.cpp
/// \brief

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/DebugInfo/CodeView/CVSymbolVisitor.h"
#include "llvm/DebugInfo/CodeView/LazyRandomTypeCollection.h"
#include "llvm/DebugInfo/CodeView/SymbolDeserializer.h"
#include "llvm/DebugInfo/CodeView/SymbolRecord.h"
#include "llvm/DebugInfo/CodeView/SymbolVisitorCallbackPipeline.h"
#include "llvm/DebugInfo/CodeView/SymbolVisitorCallbacks.h"
#include "llvm/DebugInfo/CodeView/TypeDumpVisitor.h"
#include "llvm/DebugInfo/PDB/Native/DbiStream.h"
#include "llvm/DebugInfo/PDB/Native/GlobalsStream.h"
#include "llvm/DebugInfo/PDB/Native/InfoStream.h"
#include "llvm/DebugInfo/PDB/Native/InputFile.h"
#include "llvm/DebugInfo/PDB/Native/ModuleDebugStream.h"
#include "llvm/DebugInfo/PDB/Native/NativeSession.h"
#include "llvm/DebugInfo/PDB/Native/PDBFile.h"
#include "llvm/DebugInfo/PDB/Native/SymbolStream.h"
#include "llvm/DebugInfo/PDB/PDB.h"

#include "revng/Model/Binary.h"
#include "revng/Model/Importer/DebugInfo/PDBImporter.h"
#include "revng/Model/Pass/AllPasses.h"
#include "revng/Model/Processing.h"
#include "revng/Model/QualifiedType.h"
#include "revng/Model/Type.h"
#include "revng/Support/Assert.h"
#include "revng/Support/Debug.h"
#include "revng/Support/MetaAddress.h"

using namespace llvm;
using namespace llvm::codeview;
using namespace llvm::object;
using namespace llvm::pdb;

static Logger<> DILogger("pdb-importer");

static llvm::cl::list<std::string> SearchPathPDB("search-path-pdb",
                                                 llvm::cl::desc("path"),
                                                 llvm::cl::ZeroOrMore,
                                                 llvm::cl::cat(MainCategory));
// Force using a specific PDB.
static llvm::cl::opt<std::string> UsePDB("use-pdb",
                                         llvm::cl::desc("Path to the PDB."),
                                         llvm::cl::cat(MainCategory));

namespace {
class PDBImporterImpl {
private:
  PDBImporter &Importer;
  DenseMap<TypeIndex, model::TypePath> ProcessedTypes;

public:
  PDBImporterImpl(PDBImporter &Importer) : Importer(Importer) {}
  void run(NativeSession &Session);

private:
  void populateTypes();
  void populateSymbolsWithTypes(NativeSession &Session);
};

/// Visitor for CodeView type streams found in PDB files. It overrides callbacks
/// (from `TypeVisitorCallbacks`) to types of interest for the revng `Model`.
/// During the traversal of the graph from PDB that represents the type system,
/// the `Types:` field of the `Model` is being populated. Since each CodeView
/// type in the PDB has unique `TypeIndex` that will be used when a symbol from
/// PDB symbol stream uses a certain type, we also keep a map of such
/// `TypeIndex` to corresponging type generated within the `Model` (it is done
/// by using `ProcessedTypes`), so it can be used when connecting functions from
/// `Model` with corresponding prototypes.
class PDBImporterTypeVisitor : public TypeVisitorCallbacks {
private:
  TupleTree<model::Binary> &Model;
  LazyRandomTypeCollection &Types;
  DenseMap<TypeIndex, model::TypePath> &ProcessedTypes;

  TypeIndex CurrentTypeIndex = TypeIndex::None();
  std::map<TypeIndex, SmallVector<DataMemberRecord, 8>> InProgressMemberTypes;
  std::map<TypeIndex, SmallVector<EnumeratorRecord, 8>>
    InProgressEnumeratorTypes;
  std::map<TypeIndex, ArgListRecord> InProgressArgumentsTypes;

  // Methods of a Class type. It references conrete MemberFunctionRecord.
  std::map<TypeIndex, SmallVector<OneMethodRecord, 8>>
    InProgressFunctionMemberTypes;
  DenseMap<TypeIndex, MemberFunctionRecord>
    InProgressConcreteFunctionMemberTypes;

public:
  PDBImporterTypeVisitor(TupleTree<model::Binary> &M,
                         LazyRandomTypeCollection &Types,
                         DenseMap<TypeIndex, model::TypePath> &ProcessedTypes) :
    TypeVisitorCallbacks(),
    Model(M),
    Types(Types),
    ProcessedTypes(ProcessedTypes) {}

  Error visitTypeBegin(CVType &Record) override;
  Error visitTypeBegin(CVType &Record, TypeIndex TI) override;

  Error visitKnownRecord(CVType &Record, ClassRecord &Class) override;
  Error
  visitKnownMember(CVMemberRecord &Record, EnumeratorRecord &Member) override;
  Error visitKnownRecord(CVType &Record, EnumRecord &Enum) override;
  Error visitKnownRecord(CVType &Record, ProcedureRecord &Proc) override;
  Error visitKnownRecord(CVType &Record, UnionRecord &Union) override;
  Error visitKnownRecord(CVType &Record, ArgListRecord &Args) override;
  Error
  visitKnownMember(CVMemberRecord &Record, DataMemberRecord &Member) override;
  Error visitKnownRecord(CVType &Record, FieldListRecord &FieldList) override;
  Error visitKnownRecord(CVType &Record, PointerRecord &Ptr) override;
  Error visitKnownRecord(CVType &Record, ModifierRecord &Modifier) override;
  Error visitKnownRecord(CVType &Record, ArrayRecord &Array) override;

  Error
  visitKnownMember(CVMemberRecord &Record, OneMethodRecord &FnMember) override;
  Error
  visitKnownRecord(CVType &CVR, MemberFunctionRecord &MemberFnRecord) override;

  std::optional<TupleTreeReference<model::Type, model::Binary>>
  getModelTypeForIndex(TypeIndex Index);
  void createPrimitiveType(TypeIndex SimpleType);
};

/// Visitor for CodeView symbol streams found in PDB files. It is being used for
/// connecting functions from `Model` to their prototypes. We assume the PDB
/// type stream was traversed before invoking this class.
class PDBImporterSymbolVisitor : public SymbolVisitorCallbacks {
private:
  TupleTree<model::Binary> &Model;
  DenseMap<TypeIndex, model::TypePath> &ProcessedTypes;

  NativeSession &Session;
  MetaAddress &ImageBase;

public:
  PDBImporterSymbolVisitor(TupleTree<model::Binary> &M,
                           DenseMap<TypeIndex, model::TypePath> &ProcessedTypes,
                           NativeSession &Session,
                           MetaAddress &ImageBase) :
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
  auto InputFile = InputFile::open(Importer.getPDBFile()->getFilePath());
  if (not InputFile) {
    revng_log(DILogger, "Unable to open PDB file " << InputFile.takeError());
    consumeError(InputFile.takeError());
    return;
  }

  auto StreamTpiOrErr = Importer.getPDBFile()->getPDBTpiStream();
  if (not StreamTpiOrErr) {
    revng_log(DILogger,
              "Unable to find TPI in PDB file: " << StreamTpiOrErr.takeError());
    consumeError(StreamTpiOrErr.takeError());
    return;
  }

  PDBImporterTypeVisitor TypeVisitor(Importer.getModel(),
                                     InputFile->types(),
                                     ProcessedTypes);
  if (auto Err = visitTypeStream(InputFile->types(), TypeVisitor)) {
    revng_log(DILogger, "Error during visiting types: " << Err);
    consumeError(std::move(Err));
  }
}

class PDBSymbolHandler {
private:
  PDBImporter &Importer;
  DenseMap<TypeIndex, model::TypePath> &ProcessedTypes;
  NativeSession &Session;
  InputFile &Input;

public:
  PDBSymbolHandler(PDBImporter &Importer,
                   DenseMap<TypeIndex, model::TypePath> &ProcessedTypes,
                   NativeSession &Session,
                   InputFile &Input) :
    Importer(Importer),
    ProcessedTypes(ProcessedTypes),
    Session(Session),
    Input(Input) {}

  Error operator()(uint32_t Modi, const SymbolGroup &SG) {
    auto ExpectedModS = getModuleDebugStream(*Importer.getPDBFile(), Modi);
    if (ExpectedModS) {
      ModuleDebugStreamRef &ModS = *ExpectedModS;

      SymbolVisitorCallbackPipeline Pipeline;
      SymbolDeserializer Deserializer(nullptr, CodeViewContainer::Pdb);
      PDBImporterSymbolVisitor SymVisitor(Importer.getModel(),
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
      consumeError(ExpectedModS.takeError());
    }

    return Error::success();
  }
};

void PDBImporterImpl::populateSymbolsWithTypes(NativeSession &Session) {
  auto InputFile = InputFile::open(Importer.getPDBFile()->getFilePath());
  if (not InputFile) {
    revng_log(DILogger, "Unable to open PDB file: " << InputFile.takeError());
    consumeError(InputFile.takeError());
    return;
  }

  LinePrinter Printer(/*Indent=*/2, false, nulls(), Filters);
  const PrintScope HeaderScope(Printer, /*IndentLevel=*/2);
  PDBSymbolHandler SymbolHandler(Importer, ProcessedTypes, Session, *InputFile);
  if (auto Err = iterateSymbolGroups(*InputFile, HeaderScope, SymbolHandler)) {
    revng_log(DILogger, "Unable to parse symbols: " << Err);
    consumeError(std::move(Err));
    return;
  }
}

void PDBImporterImpl::run(NativeSession &Session) {
  populateTypes();
  populateSymbolsWithTypes(Session);

  deduplicateEquivalentTypes(Importer.getModel());
  promoteOriginalName(Importer.getModel());
  purgeUnnamedAndUnreachableTypes(Importer.getModel());
  revng_assert(Importer.getModel()->verify(true));
}

void PDBImporter::loadDataFromPDB(std::string PDBFileName) {
  auto Err = loadDataForPDB(PDB_ReaderType::Native, PDBFileName, Session);
  if (not Err) {
    TheNativeSession = static_cast<NativeSession *>(Session.get());
    // TODO: We are using the static_cast due to lack of an LLVM RTTI
    // support for this. Once it is improved in LLVM, we should avoid this.
    auto SessionLoadAddress = Session->getLoadAddress();
    auto NativeSessionLoadAddress = TheNativeSession->getLoadAddress();
    revng_assert(SessionLoadAddress == NativeSessionLoadAddress);

    ThePDBFile = &TheNativeSession->getPDBFile();
  } else {
    revng_log(DILogger, "Unable to read PDB file: " << Err);
    consumeError(std::move(Err));
  }
}

void PDBImporter::import(const COFFObjectFile &TheBinary) {
  // Parse debug info and populate types to Model.
  const codeview::DebugInfo *DebugInfo;
  StringRef PDBFilePath;

  auto EC = TheBinary.getDebugPDBInfo(DebugInfo, PDBFilePath);
  if (not EC and DebugInfo != nullptr and not PDBFilePath.empty()) {
    if (llvm::sys::fs::exists(PDBFilePath)) {
      loadDataFromPDB(PDBFilePath.str());
    } else {
      // Usualy the PDB files will be generated on a different machine,
      // so the location read from the debug directory wont be up to date. At
      // first try to find it in the current `.` dir.
      auto PDBFileNameOnly = PDBFilePath.substr(PDBFilePath.find_last_of('\\')
                                                + 1);
      if (llvm::sys::fs::exists(PDBFileNameOnly.str())) {
        loadDataFromPDB(PDBFileNameOnly.str());
      } else if (not UsePDB.empty()) {
        // Sometimes we may rename a PDB file, so we can force using that one.
        loadDataFromPDB(UsePDB);
      } else if (SearchPathPDB.size() > 0) {
        // Otherwise, try to find the PDB in user defined paths.
        for (const std::string &Path : SearchPathPDB) {
          std::string UserDefinedPDBFilePath = Path + PDBFileNameOnly.str();
          if (llvm::sys::fs::exists(UserDefinedPDBFilePath)) {
            loadDataFromPDB(UserDefinedPDBFilePath);
            break;
          }
        }
      }
    }
  } else {
    revng_log(DILogger, "Unable to find PDB path in the binary.");
    if (EC) {
      revng_log(DILogger, "Unexpected debug directory: " << EC);
      consumeError(std::move(EC));
    }
    return;
  }

  if (not ThePDBFile) {
    revng_log(DILogger, "Unable to find PDB file.");
    return;
  } else {
    // According to llvm/docs/PDB/PdbStream.rst, the `Signature` was never
    // used the way as it was the initial idea. Instead, GUID is a 128-bit
    // identifier guaranteed to be unique ID for both executable and
    // corresponding PDB.
    codeview::GUID GUIDFromExe;
    llvm::copy(DebugInfo->PDB70.Signature, std::begin(GUIDFromExe.Guid));
    auto PDBInfoStrm = ThePDBFile->getPDBInfoStream();
    if (!PDBInfoStrm)
      revng_log(DILogger, "No PDB Info stream found.");
    else {
      codeview::GUID GUIDFromPDBFIle = PDBInfoStrm->getGuid();
      if (GUIDFromExe != GUIDFromPDBFIle)
        revng_log(DILogger, "Signatures from exe and PDB file mismatch.");
    }
  }

  PDBImporterImpl ModelCreator(*this);
  ModelCreator.run(*TheNativeSession);
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
static uint32_t getPointerSizeInBytes(codeview::PointerKind K) {
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
  using namespace model;
  auto TypeTypedef = makeType<TypedefType>();

  TypeIndex ReferencedType = Ptr.getReferentType();
  auto ReferencedTypeFromModel = getModelTypeForIndex(ReferencedType);
  if (!ReferencedTypeFromModel) {
    revng_log(DILogger,
              "LF_POINTER: Unknown referenced type "
                << ReferencedType.getIndex());
  } else {
    auto PointerSize = getPointerSizeInBytes(Ptr.getPointerKind());
    std::vector<Qualifier> Qualifiers{ Qualifier::createPointer(PointerSize) };
    QualifiedType TheUnderlyingType(*ReferencedTypeFromModel, Qualifiers);

    auto TheTypeTypeDef = cast<TypedefType>(TypeTypedef.get());
    TheTypeTypeDef->UnderlyingType() = TheUnderlyingType;

    auto TypePath = Model->recordNewType(std::move(TypeTypedef));
    ProcessedTypes[CurrentTypeIndex] = TypePath;
  }

  return Error::success();
}

// Parse LF_ARRAY.
Error PDBImporterTypeVisitor::visitKnownRecord(CVType &Record,
                                               ArrayRecord &Array) {
  using namespace model;
  auto TypeTypedef = makeType<TypedefType>();

  TypeIndex ElementType = Array.getElementType();

  auto ElementTypeFromModel = getModelTypeForIndex(ElementType);
  if (!ElementTypeFromModel) {
    revng_log(DILogger,
              "LF_ARRAY: Unknown element type " << ElementType.getIndex());
  } else {
    auto MaybeSize = ElementTypeFromModel->get()->size();
    if (not MaybeSize or *MaybeSize == 0 or Array.getSize() == 0) {
      revng_log(DILogger, "Skipping 0-sized array.");
      return Error::success();
    }

    const uint64_t ArraySize = Array.getSize() / *MaybeSize;
    std::vector<Qualifier> Qualifiers{ Qualifier::createArray(ArraySize) };
    QualifiedType TheUnderlyingType(*ElementTypeFromModel, Qualifiers);

    auto TheTypeTypeDef = cast<TypedefType>(TypeTypedef.get());
    TheTypeTypeDef->UnderlyingType() = TheUnderlyingType;

    auto TypePath = Model->recordNewType(std::move(TypeTypedef));
    ProcessedTypes[CurrentTypeIndex] = TypePath;
  }

  return Error::success();
}

// Parse LF_MODIFIER.
Error PDBImporterTypeVisitor::visitKnownRecord(CVType &Record,
                                               ModifierRecord &Modifier) {
  auto TypeTypedef = model::makeType<model::TypedefType>();
  TypeIndex ReferencedType = Modifier.getModifiedType();

  auto ReferencedTypeFromModel = getModelTypeForIndex(ReferencedType);
  if (!ReferencedTypeFromModel) {
    revng_log(DILogger,
              "LF_MODIFIER: Unknown referenced type "
                << ReferencedType.getIndex());
  } else {
    std::vector<model::Qualifier> Qualifiers;
    auto HasConst = Modifier.getModifiers() & ModifierOptions::Const;
    if (HasConst != ModifierOptions::None)
      Qualifiers.push_back(model::Qualifier::Qualifier::createConst());

    if (Qualifiers.size() != 0) {
      model::QualifiedType TheUnderlyingType(*ReferencedTypeFromModel,
                                             Qualifiers);

      auto TheTypeTypeDef = cast<model::TypedefType>(TypeTypedef.get());
      TheTypeTypeDef->UnderlyingType() = TheUnderlyingType;

      auto TypePath = Model->recordNewType(std::move(TypeTypedef));
      ProcessedTypes[CurrentTypeIndex] = TypePath;
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
  // 0-sized structs are typedefed to void.
  if (not Class.getSize()) {
    auto TypeTypedef = makeType<TypedefType>();
    TypeTypedef->OriginalName() = Class.getName();

    using Values = model::PrimitiveTypeKind::Values;
    QualifiedType TheUnderlyingType(Model->getPrimitiveType(Values::Void, 0),
                                    {});
    auto TheTypeTypeDef = cast<TypedefType>(TypeTypedef.get());
    TheTypeTypeDef->UnderlyingType() = TheUnderlyingType;

    auto TypePath = Model->recordNewType(std::move(TypeTypedef));
    ProcessedTypes[CurrentTypeIndex] = TypePath;

    return Error::success();
  }

  TypeIndex FieldsTypeIndex = Class.getFieldList();
  if (InProgressMemberTypes.count(FieldsTypeIndex)) {
    auto NewType = makeType<model::StructType>();
    NewType->OriginalName() = Class.getName();
    auto Struct = cast<model::StructType>(NewType.get());

    Struct->Size() = Class.getSize();
    auto &TheFields = InProgressMemberTypes[FieldsTypeIndex];
    uint64_t MaxOffset = 0;

    for (const auto &Field : TheFields) {
      // Create new field.
      uint64_t Offset = Field.getFieldOffset();
      auto FiledTypeFromModel = getModelTypeForIndex(Field.getType());
      if (!FiledTypeFromModel) {
        revng_log(DILogger,
                  "LF_STRUCTURE: Unknown field type "
                    << Field.getType().getIndex());
      } else {
        auto MaybeSize = FiledTypeFromModel->get()->size();
        uint64_t Size = MaybeSize.value_or(0);
        if (Size == 0) {
          // Skip 0-sized field.
          revng_log(DILogger, "Skipping 0-sized struct field.");
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

        // TODO: How is this posible?
        // Trigers:
        // `Last field ends outside the struct`.
        if (CurrFieldOffset > Struct->Size()) {
          revng_log(DILogger,
                    "Skipping struct field that is outside the struct.");
          continue;
        }

        auto &FieldType = Struct->Fields()[Offset];
        FieldType.OriginalName() = Field.getName().str();
        QualifiedType TheUnderlyingType(*FiledTypeFromModel, {});
        FieldType.Type() = TheUnderlyingType;
      }
    }
    auto TypePath = Model->recordNewType(std::move(NewType));
    ProcessedTypes[CurrentTypeIndex] = TypePath;
  }

  // Process methods. Create C-like function prototype for it.
  if (InProgressFunctionMemberTypes.count(FieldsTypeIndex)) {
    auto &TheFunctions = InProgressFunctionMemberTypes[FieldsTypeIndex];
    for (auto &Function : TheFunctions) {
      TypeIndex FnTypeIndex = Function.getType();
      if (not InProgressConcreteFunctionMemberTypes.count(FnTypeIndex))
        continue;

      // Get the proper LF_MFUNCTION.
      auto &MemberFunction = InProgressConcreteFunctionMemberTypes[FnTypeIndex];
      TypeIndex ReturnTypeIndex = MemberFunction.ReturnType;
      auto ReferencedTypeFromModel = getModelTypeForIndex(ReturnTypeIndex);
      if (not ReferencedTypeFromModel) {
        revng_log(DILogger,
                  "LF_MFUNCTION: Unknown return type "
                    << ReturnTypeIndex.getIndex());
        continue;
      }

      auto NewType = makeType<CABIFunctionType>();
      auto TypeFunction = cast<CABIFunctionType>(NewType.get());
      TypeFunction->ABI() = Model->DefaultABI();

      QualifiedType TheReturnType(*ReferencedTypeFromModel, {});
      TypeFunction->ReturnType() = TheReturnType;

      TypeIndex ArgListTyIndex = MemberFunction.getArgumentList();
      revng_assert(InProgressArgumentsTypes.count(ArgListTyIndex));
      auto ArgList = InProgressArgumentsTypes[ArgListTyIndex];

      auto Indices = ArgList.getIndices();
      uint32_t Size = Indices.size();
      uint32_t Index = 0;

      // Add `this` pointer as an argument if the method is not marked
      // as `static` or `friend`.
      if (Function.getMethodKind() != MethodKind::Static
          and Function.getMethodKind() != MethodKind::Friend
          and ProcessedTypes.count(CurrentTypeIndex)) {
        auto MaybeSize = ProcessedTypes[CurrentTypeIndex].get()->size();
        if (MaybeSize and *MaybeSize != 0) {
          Argument &NewArgument = TypeFunction->Arguments()[Index];
          auto PointerSize = getPointerSize(Model->Architecture());
          QualifiedType TheType(ProcessedTypes[CurrentTypeIndex],
                                { Qualifier::createPointer(PointerSize) });
          NewArgument.Type() = TheType;
          ++Index;
        } else {
          revng_log(DILogger, "Skipping 0-sized argument.");
        }
      }

      for (uint32_t I = 0; I < Size; ++I) {
        TypeIndex ArgumentTypeIndex = Indices[I];
        auto ArgumentTypeFromModel = getModelTypeForIndex(ArgumentTypeIndex);
        if (not ArgumentTypeFromModel) {
          revng_log(DILogger,
                    "LF_MFUNCTION: Unknown arg type "
                      << ArgumentTypeIndex.getIndex());
        } else {
          auto MaybeSize = ArgumentTypeFromModel->get()->size();
          uint64_t Size = MaybeSize.value_or(0);
          if (Size == 0) {
            // Skip 0-sized type.
            revng_log(DILogger, "Skipping 0-sized argument.");
            continue;
          }

          Argument &NewArgument = TypeFunction->Arguments()[Index];

          QualifiedType TheUnderlyingType(*ArgumentTypeFromModel, {});
          NewArgument.Type() = TheUnderlyingType;
          ++Index;
        }
      }

      auto TypePath = Model->recordNewType(std::move(NewType));
      ProcessedTypes[FnTypeIndex] = TypePath;
    }
  }

  return Error::success();
}

// LF_ENUM (TPI)
Error PDBImporterTypeVisitor::visitKnownRecord(CVType &Record,
                                               EnumRecord &Enum) {
  TypeIndex FieldsTypeIndex = Enum.getFieldList();
  auto NewType = model::makeType<model::EnumType>();
  NewType->OriginalName() = Enum.getName();

  TypeIndex UnderlyingTypeIndex = Enum.getUnderlyingType();
  auto UnderlynigTypeFromModel = getModelTypeForIndex(UnderlyingTypeIndex);
  if (not UnderlynigTypeFromModel) {
    revng_log(DILogger,
              "LF_ENUM: Unknown underlying type "
                << UnderlyingTypeIndex.getIndex());
    return Error::success();
  }

  model::QualifiedType TheUnderlyingType(*UnderlynigTypeFromModel, {});
  auto TypeEnum = cast<model::EnumType>(NewType.get());
  TypeEnum->UnderlyingType() = TheUnderlyingType;

  auto &TheFields = InProgressEnumeratorTypes[FieldsTypeIndex];
  if (TheFields.empty())
    return Error::success();

  for (const auto &Entry : TheFields) {
    auto &EnumEntry = TypeEnum->Entries()[Entry.getValue().getExtValue()];
    EnumEntry.OriginalName() = Entry.getName().str();
  }

  auto TypePath = Model->recordNewType(std::move(NewType));
  ProcessedTypes[CurrentTypeIndex] = TypePath;

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
    case CallingConvention::ClrCall:
    case CallingConvention::NearPascal:
    case CallingConvention::NearVector:
      return model::ABI::Microsoft_x86_64;
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
      return model::ABI::Microsoft_x86_clrcall;
    case CallingConvention::NearPascal:
      return model::ABI::Pascal_x86;
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
             and CallConv == CallingConvention::ArmCall) {
    return model::ABI::AAPCS64;
  } else {
    revng_abort();
  }
}

// LF_PROCEDURE (TPI)
Error PDBImporterTypeVisitor::visitKnownRecord(CVType &Record,
                                               ProcedureRecord &Proc) {
  TypeIndex ReturnTypeIndex = Proc.ReturnType;
  auto ReturnTypeFromModel = getModelTypeForIndex(ReturnTypeIndex);
  if (not ReturnTypeFromModel) {
    revng_log(DILogger,
              "LF_PROCEDURE: Unknown return type "
                << ReturnTypeIndex.getIndex());
  } else {
    auto NewType = model::makeType<model::CABIFunctionType>();
    auto TypeFunction = cast<model::CABIFunctionType>(NewType.get());
    TypeFunction->ABI() = getMicrosoftABI(Proc.getCallConv(),
                                          Model->Architecture());

    model::QualifiedType TheReturnType(*ReturnTypeFromModel, {});
    TypeFunction->ReturnType() = TheReturnType;

    TypeIndex ArgListTyIndex = Proc.getArgumentList();
    auto ArgumentList = InProgressArgumentsTypes[ArgListTyIndex];

    auto Indices = ArgumentList.getIndices();
    uint32_t Size = Indices.size();
    uint32_t Index = 0;
    for (uint32_t I = 0; I < Size; ++I) {
      TypeIndex ArgumentTypeIndex = Indices[I];
      auto ArgumentTypeFromModel = getModelTypeForIndex(ArgumentTypeIndex);
      if (not ArgumentTypeFromModel) {
        revng_log(DILogger,
                  "LF_PROCEDURE: Unknown argument type "
                    << ArgumentTypeIndex.getIndex());
      } else {
        auto MaybeSize = ArgumentTypeFromModel->get()->size();
        uint64_t Size = MaybeSize.value_or(0);
        if (Size == 0) {
          // Skip 0-sized type.
          revng_log(DILogger, "Skipping 0-sized argument.");
          continue;
        }

        model::Argument &NewArgument = TypeFunction->Arguments()[Index];
        model::QualifiedType TheArgumentType(*ArgumentTypeFromModel, {});

        NewArgument.Type() = TheArgumentType;
        ++Index;
      }
    }

    auto TypePath = Model->recordNewType(std::move(NewType));
    ProcessedTypes[CurrentTypeIndex] = TypePath;
  }

  return Error::success();
}

// LF_UNION (TPI)
Error PDBImporterTypeVisitor::visitKnownRecord(CVType &Record,
                                               UnionRecord &Union) {
  TypeIndex FieldsTypeIndex = Union.getFieldList();
  auto NewType = model::makeType<model::UnionType>();
  NewType->OriginalName() = Union.getName().str();

  uint64_t Index = 0;
  auto &TheFields = InProgressMemberTypes[FieldsTypeIndex];

  // Handle an empty union, similar to 0-sized structs.
  // Typedef it to void.
  if (TheFields.size() == 0) {
    auto TypeTypedef = model::makeType<model::TypedefType>();
    TypeTypedef->OriginalName() = Union.getName().str();

    auto TheTypeTypeDef = cast<model::TypedefType>(TypeTypedef.get());
    using Values = model::PrimitiveTypeKind::Values;
    auto ThePrimitiveType = Model->getPrimitiveType(Values::Void, 0);
    model::QualifiedType TheUnderlyingType(ThePrimitiveType, {});
    TheTypeTypeDef->UnderlyingType() = TheUnderlyingType;

    auto TypePath = Model->recordNewType(std::move(TypeTypedef));
    ProcessedTypes[CurrentTypeIndex] = TypePath;

    return Error::success();
  }

  bool GeneratedOneFieldAtleast = false;
  for (const auto &Field : TheFields) {
    // Create new field.
    uint64_t Offset = Field.getFieldOffset();
    auto FiledTypeFromModel = getModelTypeForIndex(Field.getType());
    if (!FiledTypeFromModel) {
      revng_log(DILogger,
                "LF_UNION: Unknown field type " << Field.getType().getIndex());
    } else {
      auto MaybeSize = FiledTypeFromModel->get()->size();
      uint64_t Size = MaybeSize.value_or(0);
      if (Size == 0) {
        // Skip 0-sized field.
        revng_log(DILogger, "Skipping 0-sized union field.");
        continue;
      }

      GeneratedOneFieldAtleast = true;
      auto TypeUnion = cast<model::UnionType>(NewType.get());
      auto &FieldType = TypeUnion->Fields()[Index];
      FieldType.OriginalName() = Field.getName().str();
      model::QualifiedType TheFieldType(*FiledTypeFromModel, {});
      FieldType.Type() = TheFieldType;

      Index++;
    }
  }

  if (GeneratedOneFieldAtleast) {
    auto TypePath = Model->recordNewType(std::move(NewType));
    ProcessedTypes[CurrentTypeIndex] = TypePath;
  }

  return Error::success();
}

Error PDBImporterTypeVisitor::visitKnownRecord(CVType &Record,
                                               ArgListRecord &Args) {
  InProgressArgumentsTypes[CurrentTypeIndex] = Args;
  return Error::success();
}

// TODO: This can go into LLVM, but thre is an ongoing review that should
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

static model::PrimitiveTypeKind::Values
codeviewSimpleTypeEncodingToModel(TypeIndex TI) {
  if (not TI.isSimple())
    return model::PrimitiveTypeKind::Invalid;

  switch (TI.getSimpleKind()) {
  case SimpleTypeKind::Void:
    return model::PrimitiveTypeKind::Void;
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
    return model::PrimitiveTypeKind::Unsigned;
  case SimpleTypeKind::SignedCharacter:
  case SimpleTypeKind::WideCharacter:
  case SimpleTypeKind::NarrowCharacter:
  case SimpleTypeKind::Character16:
  case SimpleTypeKind::Character32:
  case SimpleTypeKind::SByte:
  case SimpleTypeKind::Int32Long:
  case SimpleTypeKind::Int32:
  case SimpleTypeKind::Int64Quad:
  case SimpleTypeKind::Int64:
  case SimpleTypeKind::Int128Oct:
  case SimpleTypeKind::Int128:
    return model::PrimitiveTypeKind::Signed;
  case SimpleTypeKind::Float16:
  case SimpleTypeKind::Float32:
  case SimpleTypeKind::Float64:
  case SimpleTypeKind::Float80:
  case SimpleTypeKind::Float128:
    return model::PrimitiveTypeKind::Float;
  default:
    return model::PrimitiveTypeKind::Invalid;
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

void PDBImporterTypeVisitor::createPrimitiveType(TypeIndex SimpleType) {
  using namespace model;
  using Values = PrimitiveTypeKind::Values;
  Values Kind = codeviewSimpleTypeEncodingToModel(SimpleType);

  // If it is a pointer of size 2, lets create a PointerOrNumber for it.
  if (isTwoBytesLongPointer(SimpleType)) {
    constexpr uint64_t MSDOS16PointerSize = 2;
    auto ModelType = Model->getPrimitiveType(PrimitiveTypeKind::PointerOrNumber,
                                             MSDOS16PointerSize);
    ProcessedTypes[SimpleType] = ModelType;
  } else if (isSixteenBytesLongPointer(SimpleType)) {
    // If it is a 128-bit long pointer, typedef it to void for now. It can be
    // represented as a `struct { pointee; offset; }` since it is how it is
    // implemented in the msvc compiler.
    auto VoidModelType = Model->getPrimitiveType(PrimitiveTypeKind::Void, 0);
    auto TypeTypedef = makeType<TypedefType>();
    auto TheTypeTypeDef = cast<TypedefType>(TypeTypedef.get());
    QualifiedType TheUnderlyingType(VoidModelType, {});
    TheTypeTypeDef->UnderlyingType() = TheUnderlyingType;
    ProcessedTypes[SimpleType] = VoidModelType;
  } else {

    auto TypeSize = getSizeinBytes(SimpleType);
    if (TypeSize and Kind != PrimitiveTypeKind::Invalid) {
      // Remember the type.
      auto PrimitiveModelType = Model->getPrimitiveType(Kind, *TypeSize);
      // If it is not a pointer `SimpleTypeIndex` will be the same as
      // `SimpleType`. In the case of pointer we have some additional bits set
      // in the TypeIndex representing the type.
      TypeIndex SimpleTypeIndex(SimpleType.getSimpleKind());
      ProcessedTypes[SimpleTypeIndex] = PrimitiveModelType;

      if (not isPointer(SimpleType))
        return;
      // Create a pointer to the primitive type.
      auto PointerSize = getPointerSizeFromPDB(SimpleType);
      if (!PointerSize) {
        revng_log(DILogger, "Invalid pointer size " << SimpleType.getIndex());
        return;
      }

      auto TypeTypedef = makeType<TypedefType>();
      auto TheTypeTypeDef = cast<TypedefType>(TypeTypedef.get());
      std::vector<Qualifier> Qualifiers;
      Qualifiers.push_back({ Qualifier::createPointer(*PointerSize) });

      QualifiedType TheUnderlyingType(PrimitiveModelType, Qualifiers);
      TheTypeTypeDef->UnderlyingType() = TheUnderlyingType;

      auto TypePath = Model->recordNewType(std::move(TypeTypedef));
      ProcessedTypes[SimpleType] = TypePath;
    } else {
      revng_log(DILogger, "Invalid simple type " << SimpleType.getIndex());
    }
  }
}

std::optional<TupleTreeReference<model::Type, model::Binary>>
PDBImporterTypeVisitor::getModelTypeForIndex(TypeIndex Index) {
  if (ProcessedTypes.count(Index) != 0)
    return ProcessedTypes[Index];

  if (Index.isSimple())
    createPrimitiveType(Index);

  if (ProcessedTypes.count(Index) != 0)
    return ProcessedTypes[Index];
  return std::nullopt;
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
  // If it is not in the .idata already, we assume it is a static symbol.
  if (not Model->ImportedDynamicFunctions().count(Proc.Name.str())) {
    uint64_t FunctionVirtualAddress = Session
                                        .getRVAFromSectOffset(Proc.Segment,
                                                              Proc.CodeOffset);
    // Relocate the symbol.
    MetaAddress FunctionAddress = ImageBase + FunctionVirtualAddress;

    if (not Model->Functions().count(FunctionAddress)) {
      model::Function &Function = Model->Functions()[FunctionAddress];
      Function.OriginalName() = Proc.Name;
      TypeIndex FunctionTypeIndex = Proc.FunctionType;
      if (ProcessedTypes.count(FunctionTypeIndex)) {
        model::QualifiedType ThePrototype(ProcessedTypes[FunctionTypeIndex],
                                          {});
        Function.Prototype() = ThePrototype.UnqualifiedType();
      }
    } else {
      auto It = Model->Functions().find(FunctionAddress);
      TypeIndex FunctionTypeIndex = Proc.FunctionType;
      if (ProcessedTypes.count(FunctionTypeIndex)) {
        model::QualifiedType ThePrototype(ProcessedTypes[FunctionTypeIndex],
                                          {});
        It->Prototype() = ThePrototype.UnqualifiedType();
      }
    }
  }

  // TODO: Handle Imported functions.

  return Error::success();
}
