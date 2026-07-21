#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <optional>

#include "revng/Clift/Clift.h"
#include "revng/Ranks/Location.h"
#include "revng/Ranks/Ranks.h"
#include "revng/Support/MetaAddress.h"

namespace clift {

inline MetaAddress getMetaAddress(clift::FunctionOp F) {
  if (auto L = pipeline::locationFromString(revng::ranks::Function,
                                            F.getHandle())) {
    auto [Key] = L->at(revng::ranks::Function);
    return Key;
  }
  return MetaAddress::invalid();
}

inline auto
getUniqueIsolatedFunction(ConstOrNot<mlir::ModuleOp> auto Module,
                          const MetaAddress &Address = MetaAddress::invalid())
  -> ConstIf<std::is_const_v<decltype(Module)>, FunctionOp> {
  using FunctionType = ConstIf<std::is_const_v<decltype(Module)>, FunctionOp>;

  std::optional<FunctionType> FoundFunction;
  std::optional<MetaAddress> FoundMetaAddress;
  Module->walk([&FoundFunction, &FoundMetaAddress](clift::FunctionOp Function) {
    if (Function.isExternal())
      return;

    MetaAddress MA = getMetaAddress(Function);
    if (MA.isValid()) {
      revng_assert(not FoundFunction.has_value());
      FoundFunction = Function;
      FoundMetaAddress = MA;
    }
  });

  revng_assert(FoundFunction.has_value());

  if (not Address.isInvalid())
    revng_assert(FoundMetaAddress == Address);

  return *FoundFunction;
}

} // namespace clift
