#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include "revng/ADT/Concepts.h"
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

template<typename T>
inline T cantFail(llvm::ErrorOr<T> Obj) {
  revng_assert(Obj);
  return std::move(Obj.get());
}

template<RangeOf<llvm::Error> T>
inline llvm::Error joinErrors(T &Container) {
  auto Size = std::ranges::distance(Container);
  revng_check(Size > 0);

  auto Iter = Container.begin();
  llvm::Error Result{ std::move(*Iter) };
  if (Size == 1) {
    return Result;
  }
  for (Iter++; Iter < Container.end(); Iter++) {
    Result = llvm::joinErrors(std::move(Result), std::move(*Iter));
  }
  return Result;
}

inline std::string unwrapError(llvm::Error &&Error) {
  revng_assert(Error);

  std::string Result;
  auto Extractor = [&Result](const llvm::StringError &Error) -> llvm::Error {
    revng_assert(Result.empty());
    Result = Error.getMessage();
    return llvm::Error::success();
  };
  auto CatchAll = [](const llvm::ErrorInfoBase &) -> llvm::Error {
    revng_abort("Unsupported error type.");
  };

  llvm::handleAllErrors(std::move(Error), Extractor, CatchAll);

  revng_assert(not Result.empty());
  return Result;
}

template<std::same_as<llvm::Error>... T>
inline llvm::Error joinErrors(llvm::Error &&Error, T &&...Errors) {
  static_assert(sizeof...(Errors) > 0);
  if constexpr (sizeof...(Errors) == 1)
    return llvm::joinErrors(std::move(Error), std::move(Errors)...);
  else
    return llvm::joinErrors(std::move(Error), joinErrors(std::move(Errors)...));
}

} // namespace revng
