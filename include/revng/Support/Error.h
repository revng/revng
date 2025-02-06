#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include "revng/Support/Assert.h"

namespace revng {

template<typename... Ts>
inline llvm::Error createError(char const *Fmt, const Ts &...Vals) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(), Fmt, Vals...);
}

inline llvm::Error createError(const llvm::Twine &S) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(), S);
}

inline void cantFail(std::error_code EC) {
  revng_assert(not EC);
}

template<std::ranges::range T>
inline llvm::Error joinErrors(T &Container) {
  auto Iter = Container.begin();
  llvm::Error Result{ std::move(*Iter) };
  if (std::distance(Container.begin(), Container.end()) == 1) {
    return Result;
  }
  for (Iter++; Iter < Container.end(); Iter++) {
    Result = llvm::joinErrors(std::move(Result), std::move(*Iter));
  }
  return Result;
}

} // namespace revng
