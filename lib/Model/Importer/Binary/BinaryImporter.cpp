//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/Object/Archive.h"
#include "llvm/Object/Binary.h"
#include "llvm/Object/COFF.h"
#include "llvm/Object/ELFObjectFile.h"
#include "llvm/Object/MachO.h"
#include "llvm/Object/MachOUniversal.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/Error.h"

#include "revng/Model/Importer/Binary/BinaryImporter.h"
#include "revng/Model/Importer/Binary/Options.h"

#include "Importers.h"

using namespace llvm;

Error importBinary(TupleTree<model::Binary> &Model,
                   llvm::object::ObjectFile &ObjectFile,
                   llvm::StringRef FullPathForExternalTools,
                   const ImporterOptions &Options,
                   model::BinaryIdentifierReference &BinaryReference) {
  using namespace llvm::object;
  using namespace model::Architecture;

  // If we have segments, we already imported a binary, bail out.
  if (Model->Segments().size() > 0)
    return Error::success();

  Model->Architecture() = fromLLVMArchitecture(ObjectFile.getArch());

  if (Model->Architecture() == model::Architecture::Invalid)
    return revng::createError("Invalid architecture");

  llvm::Error Result = Error::success();
  revng_check(not Result);
  if (auto *TheBinary = dyn_cast<ELFObjectFileBase>(&ObjectFile)) {
    ELFBinary Binary(*TheBinary,
                     FullPathForExternalTools.str(),
                     BinaryReference);
    Result = importELF(Model, Binary, Options);
  } else if (auto *TheBinary = dyn_cast<COFFObjectFile>(&ObjectFile)) {
    Model->OperatingSystem() = model::OperatingSystem::Windows;
    COFFBinary Binary(*TheBinary,
                      FullPathForExternalTools.str(),
                      BinaryReference);
    Result = importPECOFF(Model, Binary, Options);
  } else if (auto *TheBinary = dyn_cast<MachOObjectFile>(&ObjectFile)) {
    Model->OperatingSystem() = model::OperatingSystem::MacOS;
    MachOBinary Binary(*TheBinary,
                       FullPathForExternalTools.str(),
                       BinaryReference);
    Result = importMachO(Model, Binary, Options);
  } else {
    return revng::createError("Unsupported binary format");
  }

  if (Result)
    return Result;

  // If we got an EntryPoint, ensure we also have Function for it
  if (Model->EntryPoint().isValid())
    Model->Functions()[Model->EntryPoint()];

  return Result;
}

Error importBinary(TupleTree<model::Binary> &Model,
                   llvm::StringRef FullPathForExternalTools,
                   const ImporterOptions &Options,
                   model::BinaryIdentifierReference &BinaryReference) {

  auto
    MaybeBuffer = llvm::MemoryBuffer::getFileOrSTDIN(FullPathForExternalTools,
                                                     false,
                                                     false);
  if (not MaybeBuffer)
    return llvm::errorCodeToError(MaybeBuffer.getError());

  auto BinaryOrError = object::createBinary(**MaybeBuffer);
  if (not BinaryOrError)
    return BinaryOrError.takeError();

  llvm::object::Binary &Binary = *BinaryOrError->get();
  if (isa<object::MachOUniversalBinary>(Binary)) {
    return revng::createError("Unsupported format: MachO universal binary.");
  } else if (isa<llvm::object::Archive>(Binary)) {
    return revng::createError("Unsupported format: archive.");
  } else if (not isa<object::ObjectFile>(Binary)) {
    return revng::createError("Unsupported format");
  }

  return importBinary(Model,
                      cast<object::ObjectFile>(Binary),
                      FullPathForExternalTools,
                      Options,
                      BinaryReference);
}
