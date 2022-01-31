#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <map>

#include "llvm/ADT/Twine.h"
#include "llvm/Support/ManagedStatic.h"

#include "revng/Model/Binary.h"

class RegisterModelPass {
public:
  using ModelPass = std::function<void(TupleTree<model::Binary> &)>;

private:
  static llvm::ManagedStatic<std::map<std::string, ModelPass>> Registry;

public:
  RegisterModelPass(const llvm::Twine &Name, ModelPass Pass) {
    bool Inserted = Registry->emplace(Name.str(), Pass).second;
    revng_assert(Inserted);
  }

public:
  static const ModelPass *get(llvm::StringRef Name) {
    auto It = Registry->find(Name.str());
    if (It == Registry->end())
      return nullptr;
    else
      return &It->second;
  }

  static auto passes() {
    return llvm::make_range(Registry->begin(), Registry->end());
  }
};

inline llvm::ManagedStatic<std::map<std::string, RegisterModelPass::ModelPass>>
  RegisterModelPass::Registry;
