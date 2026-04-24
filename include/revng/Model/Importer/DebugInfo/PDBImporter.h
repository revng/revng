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

struct ImporterOptions;

class PDBImporter : public BinaryImporterHelper {
public:
  using AddressWhitelist = std::set<uint64_t>;

private:
  MetaAddress ImageBase;
  llvm::pdb::PDBFile *ThePDBFile = nullptr;
  llvm::pdb::NativeSession *TheNativeSession = nullptr;
  std::unique_ptr<llvm::pdb::IPDBSession> Session;
  std::optional<llvm::codeview::GUID> ExpectedGUID;
  const AddressWhitelist *FunctionWhitelist = nullptr;

public:
  PDBImporter(TupleTree<model::Binary> &Model,
              const MetaAddress &ImageBase,
              const std::optional<AddressWhitelist> &FunctionWhitelist);

  PDBImporter(TupleTree<model::Binary> &Model,
              const std::optional<AddressWhitelist> &FunctionWhitelist) :
    PDBImporter(Model,
                MetaAddress::fromGeneric(Model->Architecture(), 0),
                FunctionWhitelist) {}

  TupleTree<model::Binary> &getModel() { return Binary; }
  const MetaAddress &getBaseAddress() { return ImageBase; }
  llvm::pdb::PDBFile *getPDBFile() { return ThePDBFile; }
  auto *getNativeSession() { return TheNativeSession; }

  void import(const revng::RootEntry *Root,
              const COFFBinary &Binary,
              const ImporterOptions &Options);

  void importPDB(llvm::StringRef PDBPath, const ImporterOptions &Options);

  bool loadDataFromPDB(llvm::StringRef PDBFileName);

  bool isFunctionAllowed(uint64_t Address) const {
    if (FunctionWhitelist == nullptr)
      return true;

    return FunctionWhitelist->contains(Address);
  }
};
