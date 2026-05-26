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
#include "llvm/Support/FileSystem.h"

#include "revng/Model/Binary.h"
#include "revng/Model/Importer/Binary/BinaryDescriptor.h"
#include "revng/Model/Importer/Binary/BinaryImporterHelper.h"
#include "revng/Support/Configuration.h"
#include "revng/Support/LDDTree.h"

struct ImporterOptions;

class PDBImporter : public BinaryImporterHelper {
private:
  MetaAddress ImageBase;
  llvm::pdb::PDBFile *ThePDBFile = nullptr;
  llvm::pdb::NativeSession *TheNativeSession = nullptr;
  std::unique_ptr<llvm::pdb::IPDBSession> Session;
  std::optional<llvm::codeview::GUID> ExpectedGUID;
  std::optional<std::map<MetaAddress, LDDTree::Symbol>> WhitelistByAddress;

public:
  PDBImporter(TupleTree<model::Binary> &Model,
              const MetaAddress &ImageBase,
              std::optional<std::map<MetaAddress, LDDTree::Symbol>> Whitelist);

  PDBImporter(TupleTree<model::Binary> &Model,
              std::optional<std::map<MetaAddress, LDDTree::Symbol>> Whitelist) :
    PDBImporter(Model,
                MetaAddress::fromGeneric(Model->Architecture(), 0),
                std::move(Whitelist)) {}

  TupleTree<model::Binary> &getModel() { return Binary; }
  const MetaAddress &getBaseAddress() { return ImageBase; }
  llvm::pdb::PDBFile *getPDBFile() { return ThePDBFile; }
  auto *getNativeSession() { return TheNativeSession; }

  void import(const revng::RootEntry *Root,
              const COFFBinary &Binary,
              const ImporterOptions &Options);

  void importPDB(llvm::StringRef PDBPath, const ImporterOptions &Options);

  bool loadDataFromPDB(llvm::StringRef PDBFileName);

  bool isFunctionAllowed(const MetaAddress &Address) const {
    if (not WhitelistByAddress.has_value())
      return true;

    return WhitelistByAddress->contains(Address);
  }
};
