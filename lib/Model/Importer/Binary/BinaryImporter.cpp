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

#include "Importers.h"

using namespace llvm;

Error importBinary(TupleTree<model::Binary> &Model,
                   llvm::object::ObjectFile &ObjectFile,
                   uint64_t PreferredBaseAddress) {
  using namespace llvm::object;
  using namespace model::Architecture;
  Model->Architecture = fromLLVMArchitecture(ObjectFile.getArch());

  if (auto *TheBinary = dyn_cast<ELFObjectFileBase>(&ObjectFile)) {
    return importELF(Model, *TheBinary, PreferredBaseAddress);
  } else if (auto *TheBinary = dyn_cast<COFFObjectFile>(&ObjectFile)) {
    return importPECOFF(Model, *TheBinary, PreferredBaseAddress);
  } else if (auto *TheBinary = dyn_cast<MachOObjectFile>(&ObjectFile)) {
    return importMachO(Model, *TheBinary, PreferredBaseAddress);
  } else {
    return createError("Unsupported binary format");
  }
}

Error importBinary(TupleTree<model::Binary> &Model,
                   llvm::StringRef Path,
                   uint64_t PreferredBaseAddress) {
  auto BinaryOrErr = object::createBinary(Path);
  if (not BinaryOrErr)
    return BinaryOrErr.takeError();

  return importBinary(Model,
                      *cast<object::ObjectFile>(BinaryOrErr->getBinary()),
                      PreferredBaseAddress);
}
