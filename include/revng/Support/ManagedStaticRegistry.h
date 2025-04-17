#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <compare>
#include <set>

#include "llvm/ADT/Twine.h"
#include "llvm/Support/ManagedStatic.h"

#include "revng/Support/Debug.h"

template<typename Type>
concept IsStaticallyRegisterable = requires(Type &&Value) {
  { Value.key() } -> std::convertible_to<llvm::StringRef>;
};

template<IsStaticallyRegisterable Registered>
class RegisterManagedStaticImpl {
private:
  struct TransparentKeyComparator {
    using is_transparent = std::true_type;

    template<typename Type>
    llvm::StringRef get(const Type &Value) const {
      return llvm::StringRef(Value);
    }
    llvm::StringRef get(const Registered &Value) const {
      return llvm::StringRef(Value.key());
    }
    template<typename LHS, typename RHS>
    bool operator()(const LHS &Left, const RHS &Right) const {
      return get(Left) < get(Right);
    }
  };
  using RegistryType = std::set<Registered, TransparentKeyComparator>;
  static inline llvm::ManagedStatic<RegistryType> Registry;

public:
  template<typename... Types>
  RegisterManagedStaticImpl(Types &&...Values) {
    auto [_, Success] = Registry->emplace(std::forward<Types>(Values)...);
    revng_assert(Success);
  }

public:
  static const Registered *get(llvm::StringRef Name) {
    if (auto It = Registry->find(Name); It == Registry->end())
      return nullptr;
    else
      return std::addressof(*It);
  }

  static const auto &values() { return std::as_const(*Registry); }
};
