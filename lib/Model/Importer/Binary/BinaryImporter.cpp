/// \file BinaryImporter.cpp
/// \brief

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/Object/COFF.h"
#include "llvm/Object/ELFObjectFile.h"
#include "llvm/Object/MachO.h"
#include "llvm/Object/ObjectFile.h"

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

  return importBinary(Model,
                      *cast<object::ObjectFile>(BinaryOrError->getBinary()),
                      Options);
}
