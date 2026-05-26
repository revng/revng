//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/StringExtras.h"
#include "llvm/DebugInfo/CodeView/GUID.h"
#include "llvm/DebugInfo/PDB/Native/InfoStream.h"
#include "llvm/DebugInfo/PDB/Native/NativeSession.h"
#include "llvm/DebugInfo/PDB/Native/PDBFile.h"
#include "llvm/DebugInfo/PDB/PDB.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/Program.h"

#include "revng/Model/Binary.h"
#include "revng/Model/Importer/Binary/Options.h"
#include "revng/Model/Importer/DebugInfo/PDBImporter.h"
#include "revng/Support/CommandLine.h"
#include "revng/Support/Debug.h"
#include "revng/Support/MetaAddress.h"
#include "revng/Support/PathList.h"
#include "revng/Support/ProgramRunner.h"

#include "ImportDebugInfoHelper.h"
#include "PDBImporterImpl.h"

using namespace llvm;
using namespace llvm::object;
using namespace llvm::pdb;

PDBImporter::PDBImporter(TupleTree<model::Binary> &Model,
                         const MetaAddress &ImageBase,
                         std::optional<std::map<MetaAddress, LDDTree::Symbol>>
                           Whitelist) :
  BinaryImporterHelper(Model,
                       ImageBase.isValid() ? ImageBase.address() : 0,
                       Log),
  ImageBase(ImageBase),
  WhitelistByAddress(std::move(Whitelist)) {

  // When we import debug info, we assume we already have parsed Segments
  processSegments();
}

bool PDBImporter::loadDataFromPDB(StringRef PDBFileName) {
  revng_log(Log, "Loading PDB " << PDBFileName);
  LoggerIndent Indent(Log);

  auto Error = loadDataForPDB(PDB_ReaderType::Native, PDBFileName, Session);
  if (Error) {
    revng_log(Log, "Warning: unable to read PDB file: " << Error);
    consumeError(std::move(Error));
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
      // TODO: emit a diagnostic message for the user.
      revng_log(Log, "Warning: error in getPDBInfoStream: " << Error);
      consumeError(std::move(Error));
      // TODO: is it correct to ignore this error?
      return true;
    }

    codeview::GUID GUIDFromPDBFile = MaybePDBInfoStream->getGuid();
    if (ExpectedGUID != GUIDFromPDBFile) {
      if (Log.isEnabled()) {
        {
          auto Stream = Log.getAsLLVMStream();
          *Stream << "Warning: signatures from import binary and PDB file "
                     "mismatch\n";
          *Stream << "  Expected: " << ExpectedGUID.value() << "\n";
          *Stream << "  PDB: " << GUIDFromPDBFile;
        }
        Log << DoLog;
      }
      return false;
    }
  }

  return true;
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

class PDBFinder {
private:
  const revng::RootEntry &Root;

public:
  struct Result {
    std::string Path;
    std::optional<llvm::codeview::GUID> MaybeGUID;
  };

private:
  PDBFinder(const revng::RootEntry &Root) : Root(Root) {}

public:
  static std::optional<Result> find(const revng::RootEntry &Root,
                                    const COFFObjectFile &TheBinary,
                                    llvm::StringRef FullPathForExternalTools) {
    PDBFinder Finder(Root);
    return Finder.findPDBFilePath(TheBinary, FullPathForExternalTools);
  }

private:
  std::optional<Result>
  findPDBFilePath(const llvm::object::COFFObjectFile &TheBinary,
                  llvm::StringRef BinaryPath);

  std::optional<std::string> getCachedPDBFilePath(std::string PDBFileID,
                                                  llvm::StringRef PDBFilePath);

  bool fileExists(const llvm::Twine &Path) const {
    using namespace llvm::sys;
    revng_assert(not Path.str().empty());
    bool Result = fs::exists(Path);
    if (not Result)
      revng_log(Log, "No file at the following path: " << Path.str());
    return Result;
  }
};

void PDBImporter::import(const revng::RootEntry *Root,
                         const COFFBinary &Binary,
                         const ImporterOptions &Options) {
  revng_assert(Root != nullptr);
  auto Result = PDBFinder::find(*Root,
                                Binary.ObjectFile,
                                Binary.FullPathForExternalTools);
  if (not Result.has_value()) {
    revng_log(Log, "Failed to find the PDB");
    return;
  }

  auto [PDBPath, ExpectedGUID] = *Result;
  this->ExpectedGUID = ExpectedGUID;
  importPDB(PDBPath, Options);
}

void PDBImporter::importPDB(StringRef PDBPath, const ImporterOptions &Options) {
  if (Options.DebugInfo == DebugInfoLevel::No)
    return;

  if (not loadDataFromPDB(PDBPath))
    return;

  PDBImporterImpl ModelCreator(*this);
  ModelCreator.run();
}

std::optional<std::string>
PDBFinder::getCachedPDBFilePath(std::string PDBFileID, StringRef PDBBaseName) {
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

static StringRef getBaseName(StringRef Path) {
  auto PositionOfLastDirectoryChar = Path.rfind("\\");
  if (PositionOfLastDirectoryChar != llvm::StringRef::npos) {
    return Path.slice(PositionOfLastDirectoryChar + 1, Path.size());
  }
  return Path;
}

static std::string convertToHostPath(StringRef WindowsPath) {
  using namespace llvm::sys::path;
  StringRef Rel = relative_path(WindowsPath, Style::windows);
  SmallString<128> Result(Rel);
  native(Result, Style::native);
  return get_separator().str() + std::string(Result);
}

std::optional<PDBFinder::Result>
PDBFinder::findPDBFilePath(const COFFObjectFile &TheBinary,
                           llvm::StringRef FullPathForExternalTools) {
  revng_log(Log, "Looking for the PDB file for " << FullPathForExternalTools);
  LoggerIndent Indent(Log);

  // Parse debug info in TheBinary
  const codeview::DebugInfo *DebugInfo = nullptr;
  std::string InternalPDBPath;
  {
    StringRef InternalPDBStringReference;
    auto EC = TheBinary.getDebugPDBInfo(DebugInfo, InternalPDBStringReference);
    if (EC) {
      // TODO: emit a diagnostic message for the user.
      revng_log(Log, "getDebugPDBInfo failed: " << EC);
      consumeError(std::move(EC));
      return std::nullopt;
    } else if (DebugInfo == nullptr) {
      revng_log(Log, "Couldn't get codeview::DebugInfo");
      return std::nullopt;
    } else {
      InternalPDBPath = InternalPDBStringReference.str();
      revng_log(Log, "InternalPDBPath: " << InternalPDBPath);
    }
  }

  // TODO: Handle PDB signature types other then PDB70, e.g. PDB20.
  if (DebugInfo->Signature.CVSignature != OMF::Signature::PDB70) {
    // TODO: emit a diagnostic message for the user.
    revng_log(Log, "A non-PDB70 signature was found, ignore.");
    return std::nullopt;
  }

  // According to llvm/docs/PDB/PdbStream.rst, the `Signature` was never
  // used the way as it was the initial idea. Instead, GUID is a 128-bit
  // identifier guaranteed to be unique ID for both executable and
  // corresponding PDB. Save the GUID for later to check the match.
  llvm::codeview::GUID ExpectedGUID;
  llvm::copy(DebugInfo->PDB70.Signature, std::begin(ExpectedGUID.Guid));

  if (InternalPDBPath.empty()) {
    // TODO: emit a diagnostic message for the user.
    revng_log(Log, "The internal PDB path is empty");
    return std::nullopt;
  }

  // If the internal PDB path exists, use that
  InternalPDBPath = convertToHostPath(InternalPDBPath);

  if (auto MaybeFullPath = Root.getExistingPath(InternalPDBPath)) {
    revng_log(Log, "Found: " << *MaybeFullPath);
    return { { *MaybeFullPath, ExpectedGUID } };
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

    if (auto MaybeFullPath = Root.getExistingPath(ResultPath)) {
      revng_log(Log, "Found: " << *MaybeFullPath);
      return { { *MaybeFullPath, ExpectedGUID } };
    }
  }

  // Try main input path
  ResultPath.clear();
  if (not InputPath.empty()) {
    llvm::sys::path::append(ResultPath,
                            llvm::sys::path::parent_path(InputPath),
                            PDBBaseName);
    if (auto MaybeFullPath = Root.getExistingPath(ResultPath)) {
      revng_log(Log, "Found: " << *MaybeFullPath);
      return { { *MaybeFullPath, ExpectedGUID } };
    }
  }

  // Compute the PDB file ID
  auto PDBFileID = formatPDBFileID(DebugInfo->PDB70.Signature,
                                   DebugInfo->PDB70.Age);

  // Check if we already fetched it from PDB servers in the past
  if (auto MaybeCachedPDBPath = getCachedPDBFilePath(PDBFileID, PDBBaseName)) {
    revng_log(Log, "Found: " << *MaybeCachedPDBPath);
    return { { *MaybeCachedPDBPath, ExpectedGUID } };
  }

  // Let's try finding it on web with the `fetch-debuginfo` tool.
  // If the `revng` cannot be found, avoid finding debug info.
  int ExitCode = runFetchDebugInfo(FullPathForExternalTools, Log.isEnabled());
  if (ExitCode != 0) {
    revng_log(Log,
              "Failed to find debug info with `revng model "
              "fetch-debuginfo`.");
    return std::nullopt;
  }

  // Try again to find the file
  if (auto MaybeCachedPDBPath = getCachedPDBFilePath(PDBFileID, PDBBaseName)) {
    revng_log(Log, "Found: " << *MaybeCachedPDBPath);
    return { { *MaybeCachedPDBPath, ExpectedGUID } };
  }

  revng_log(Log, "Not found");
  return std::nullopt;
}
