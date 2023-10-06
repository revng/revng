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
  Model->Architecture() = fromLLVMArchitecture(ObjectFile.getArch());

  if (Model->Architecture() == model::Architecture::Invalid)
    return createError("Invalid architecture");

  if (auto *TheBinary = dyn_cast<ELFObjectFileBase>(&ObjectFile))
    return importELF(Model, *TheBinary, Options);
  else if (auto *TheBinary = dyn_cast<COFFObjectFile>(&ObjectFile))
    return importPECOFF(Model, *TheBinary, Options);
  else if (auto *TheBinary = dyn_cast<MachOObjectFile>(&ObjectFile))
    return importMachO(Model, *TheBinary, Options);
  else
    return createError("Unsupported binary format");
}

Error importBinary(TupleTree<model::Binary> &Model,
                   llvm::StringRef Path,
                   const ImporterOptions &Options) {
  auto BinaryOrError = object::createBinary(Path);
  if (not BinaryOrError)
    return BinaryOrError.takeError();

  auto *Binary = BinaryOrError->getBinary();
  if (isa<object::MachOUniversalBinary>(Binary)) {
    return createStringError(inconvertibleErrorCode(),
                             "Unsupported format: MachO universal binary.");
  } else if (isa<llvm::object::Archive>(Binary)) {
    return createStringError(inconvertibleErrorCode(),
                             "Unsupported format: archive.");
  } else if (not isa<object::ObjectFile>(Binary)) {
    return createStringError(inconvertibleErrorCode(), "Unsupported format");
  }

  return importBinary(Model, *cast<object::ObjectFile>(Binary), Options);
}
