/// \file BinaryImporter.cpp

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
                   const ImporterOptions &Options) {
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
  if (auto *TheBinary = dyn_cast<ELFObjectFileBase>(&ObjectFile))
    Result = importELF(Model, *TheBinary, Options);
  else if (auto *TheBinary = dyn_cast<COFFObjectFile>(&ObjectFile))
    Result = importPECOFF(Model, *TheBinary, Options);
  else if (auto *TheBinary = dyn_cast<MachOObjectFile>(&ObjectFile))
    Result = importMachO(Model, *TheBinary, Options);
  else
    return revng::createError("Unsupported binary format");

  if (Result)
    return Result;

  // If we got an EntryPoint, ensure we also have Function for it
  if (Model->EntryPoint().isValid())
    Model->Functions()[Model->EntryPoint()];

  return Result;
}

Error importBinary(TupleTree<model::Binary> &Model,
                   llvm::StringRef Path,
                   const ImporterOptions &Options) {
  auto BinaryOrError = object::createBinary(Path);
  if (not BinaryOrError)
    return BinaryOrError.takeError();

  auto *Binary = BinaryOrError->getBinary();
  if (isa<object::MachOUniversalBinary>(Binary)) {
    return revng::createError("Unsupported format: MachO universal binary.");
  } else if (isa<llvm::object::Archive>(Binary)) {
    return revng::createError("Unsupported format: archive.");
  } else if (not isa<object::ObjectFile>(Binary)) {
    return revng::createError("Unsupported format");
  }

  return importBinary(Model, *cast<object::ObjectFile>(Binary), Options);
}
