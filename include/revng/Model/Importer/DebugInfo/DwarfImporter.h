#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/Object/Binary.h"

#include "revng/Model/Binary.h"
#include "revng/Model/Importer/Binary/BinaryDescriptor.h"
#include "revng/Support/Configuration.h"

struct ImporterOptions;

class DwarfImporter {
public:
  using AddressWhitelist = std::set<uint64_t>;

private:
  TupleTree<model::Binary> &Model;
  std::vector<std::string> LoadedFiles;
  using DwarfID = std::pair<size_t, size_t>;
  std::map<DwarfID, model::UpcastableType> DwarfToModel;
  const AddressWhitelist *FunctionWhitelist = nullptr;

public:
  DwarfImporter(TupleTree<model::Binary> &Model,
                const std::optional<AddressWhitelist> &FunctionWhitelist) :
    Model(Model),
    FunctionWhitelist(FunctionWhitelist.has_value() ?
                        &*FunctionWhitelist :
                        static_cast<const AddressWhitelist *>(nullptr)) {}

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

  bool isFunctionAllowed(uint64_t Address) const {
    if (FunctionWhitelist == nullptr)
      return true;

    return FunctionWhitelist->contains(Address);
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
