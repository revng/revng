#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/Object/Binary.h"

#include "revng/Model/Binary.h"

struct ImporterOptions;

class DwarfImporter {
private:
  TupleTree<model::Binary> &Model;
  std::vector<std::string> LoadedFiles;
  using DwarfID = std::pair<size_t, size_t>;
  std::map<DwarfID, model::UpcastableType> DwarfToModel;

public:
  DwarfImporter(TupleTree<model::Binary> &Model) : Model(Model) {}

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

public:
  void import(llvm::StringRef FileName, const ImporterOptions &Options);

private:
  void import(const llvm::object::Binary &TheBinary,
              llvm::StringRef FileName,
              uint64_t PreferredBaseAddress);
};
