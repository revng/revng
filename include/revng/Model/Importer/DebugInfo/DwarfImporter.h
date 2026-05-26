#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/Object/Binary.h"

#include "revng/Model/Binary.h"
#include "revng/Model/Importer/Binary/BinaryDescriptor.h"
#include "revng/Support/Configuration.h"
#include "revng/Support/LDDTree.h"

struct ImporterOptions;

class DwarfImporter {
private:
  TupleTree<model::Binary> &Model;
  std::vector<std::string> LoadedFiles;
  using DwarfID = std::pair<size_t, size_t>;
  std::map<DwarfID, model::UpcastableType> DwarfToModel;

  /// When unset the importer accepts every subprogram; otherwise only those
  /// whose MetaAddress appears in the map are kept. Keys carry the code type
  /// (Code_arm vs Code_arm_thumb, etc.), so the lookup is exact and
  /// architecture-aware.
  std::optional<std::map<MetaAddress, LDDTree::Symbol>> WhitelistByAddress;

public:
  DwarfImporter(TupleTree<model::Binary> &Model,
                std::optional<std::map<MetaAddress, LDDTree::Symbol>>
                  Whitelist) :
    Model(Model), WhitelistByAddress(std::move(Whitelist)) {}

public:
  model::UpcastableType findType(DwarfID ID) {
    auto It = DwarfToModel.find(ID);
    return It != DwarfToModel.end() ? It->second.copy() :
                                      model::UpcastableType::empty();
  }

  model::UpcastableType &recordType(DwarfID ID,
                                    model::UpcastableType &&NewType) {
    revng_assert(!DwarfToModel.contains(ID));
    return DwarfToModel.insert({ ID, std::move(NewType) }).first->second;
  }

  TupleTree<model::Binary> &getModel() { return Model; }

  bool isFunctionAllowed(const MetaAddress &Address) const {
    if (not WhitelistByAddress.has_value())
      return true;

    return WhitelistByAddress->contains(Address);
  }

  bool isIfunc(const MetaAddress &Address) const {
    if (not WhitelistByAddress.has_value())
      return false;

    auto It = WhitelistByAddress->find(Address);
    if (It == WhitelistByAddress->end())
      return false;
    return It->second.IsIfunc;
  }

public:
  size_t import(llvm::StringRef FileName, const ImporterOptions &Options);

  /// \p Root Optional.
  size_t import(const revng::RootEntry *Root,
                const ELFBinary &Binary,
                const ImporterOptions &Options);

private:
  // \return the index the imported Dwarf file, use this for DwarfID.
  size_t import(const llvm::object::Binary &TheBinary,
                llvm::StringRef CanonicalPath,
                uint64_t PreferredBaseAddress,
                size_t AltIndex);
};
