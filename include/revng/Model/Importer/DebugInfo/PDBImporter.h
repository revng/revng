#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/DebugInfo/CodeView/LazyRandomTypeCollection.h"
#include "llvm/DebugInfo/PDB/Native/InputFile.h"
#include "llvm/DebugInfo/PDB/Native/NativeSession.h"
#include "llvm/DebugInfo/PDB/Native/PDBFile.h"
#include "llvm/DebugInfo/PDB/PDB.h"
#include "llvm/Object/Binary.h"
#include "llvm/Object/COFF.h"

#include "revng/Model/Binary.h"
#include "revng/Model/Importer/Binary/BinaryImporterHelper.h"

struct ImporterOptions;

class PDBImporter : public BinaryImporterHelper {
private:
  TupleTree<model::Binary> &Model;
  MetaAddress ImageBase;
  llvm::pdb::PDBFile *ThePDBFile = nullptr;
  llvm::pdb::NativeSession *TheNativeSession = nullptr;
  std::unique_ptr<llvm::pdb::IPDBSession> Session;
  std::optional<llvm::codeview::GUID> ExpectedGUID;

public:
  PDBImporter(TupleTree<model::Binary> &Model, MetaAddress ImageBase);

  TupleTree<model::Binary> &getModel() { return Model; }
  MetaAddress &getBaseAddress() { return ImageBase; }
  llvm::pdb::PDBFile *getPDBFile() { return ThePDBFile; }

  void import(const llvm::object::COFFObjectFile &TheBinary,
              const ImporterOptions &Options);

  bool loadDataFromPDB(llvm::StringRef PDBFileName);

  std::optional<std::string>
  getPDBFilePath(const llvm::object::COFFObjectFile &TheBinary);

  std::optional<std::string> getCachedPDBFilePath(std::string PDBFileID,
                                                  llvm::StringRef PDBFilePath);
};
