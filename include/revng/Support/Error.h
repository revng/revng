#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

namespace revng {

template<typename... Ts>
inline llvm::Error createError(char const *Fmt, const Ts &...Vals) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(), Fmt, Vals...);
}

inline llvm::Error createError(const llvm::Twine &S) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(), S);
}

} // namespace revng
