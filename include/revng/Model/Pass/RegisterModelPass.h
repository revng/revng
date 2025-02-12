#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <compare>
#include <set>

#include "llvm/ADT/Twine.h"
#include "llvm/Support/ManagedStatic.h"

#include "revng/Model/Binary.h"

class RegisterModelPass {
public:
  using ModelPass = std::function<void(TupleTree<model::Binary> &)>;

private:
  struct Registered {
    std::string Name;
    std::string Description;
    ModelPass Pass;
  };
  struct TransparentNameComparator {
    using is_transparent = std::true_type;

    template<typename Type>
    llvm::StringRef get(const Type &Value) const {
      return llvm::StringRef(Value);
    }
    llvm::StringRef get(const Registered &Value) const {
      return llvm::StringRef(Value.Name);
    }
    template<typename LHS, typename RHS>
    bool operator()(const LHS &Left, const RHS &Right) const {
      return get(Left) < get(Right);
    }
  };
  using RegistryType = std::set<Registered, TransparentNameComparator>;
  static llvm::ManagedStatic<RegistryType> Registry;

public:
  RegisterModelPass(const llvm::Twine &Name,
                    const llvm::Twine &Description,
                    ModelPass Pass) {
    Registered RegisteredPass{ Name.str(), Description.str(), Pass };
    auto &&[_, Success] = Registry->emplace(std::move(RegisteredPass));
    revng_assert(Success);
  }

public:
  static const ModelPass *get(llvm::StringRef Name) {
    if (auto It = Registry->find(Name); It == Registry->end())
      return nullptr;
    else
      return &It->Pass;
  }

  static auto passes() {
    return llvm::make_range(Registry->begin(), Registry->end());
  }
};

inline llvm::ManagedStatic<RegisterModelPass::RegistryType>
  RegisterModelPass::Registry;
