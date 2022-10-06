#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <map>
#include <optional>
#include <set>
#include <type_traits>

#include "llvm/ADT/Twine.h"
#include "llvm/Support/raw_ostream.h"

#include "revng/Support/Assert.h"
#include "revng/Support/Debug.h"
#include "revng/TupleTree/TupleTree.h"

inline Logger<> ModelVerifyLogger("model-verify");

namespace model {
class Type;

class VerifyHelper {
private:
  std::set<const model::Type *> VerifiedCache;
  std::map<const model::Type *, uint64_t> SizeCache;
  std::map<const model::Type *, uint64_t> AlignmentCache;
  std::set<const model::Type *> InProgress;
  bool AssertOnFail = false;

public:
  VerifyHelper() = default;
  VerifyHelper(bool AssertOnFail) : AssertOnFail(AssertOnFail) {}

  ~VerifyHelper() { revng_assert(InProgress.size() == 0); }

public:
  void setVerified(const model::Type *T) {
    revng_assert(not isVerified(T));
    VerifiedCache.insert(T);
  }

  bool isVerified(const model::Type *T) const {
    return VerifiedCache.count(T) != 0;
  }

public:
  bool isVerificationInProgress(const model::Type *T) const {
    return InProgress.count(T) != 0;
  }

  void verificationInProgress(const model::Type *T) {
    revng_assert(not isVerificationInProgress(T));
    revng_assert(not isVerified(T));
    InProgress.insert(T);
  }

  void verificationCompleted(const model::Type *T) {
    revng_assert(isVerificationInProgress(T));
    InProgress.erase(T);
  }

public:
  void setSize(const model::Type *T, uint64_t Size) {
    revng_assert(not size(T));
    SizeCache[T] = Size;
  }

  std::optional<uint64_t> size(const model::Type *T) {
    auto It = SizeCache.find(T);
    if (It != SizeCache.end())
      return It->second;
    else
      return std::nullopt;
  }

public:
  void setAlignment(const model::Type *T, uint64_t NewValue) {
    revng_assert(not alignment(T));
    AlignmentCache[T] = NewValue;
  }

  std::optional<uint64_t> alignment(const model::Type *T) {
    auto It = AlignmentCache.find(T);
    if (It != AlignmentCache.end())
      return It->second;
    else
      return std::nullopt;
  }

public:
  bool maybeFail(bool Result) const { return maybeFail(Result, {}); }

  bool maybeFail(bool Result, const llvm::Twine &Reason) const {
    if (AssertOnFail and not Result) {
      revng_abort(Reason.str().c_str());
    } else {
      return Result;
    }
  }

  template<typename T>
  bool maybeFail(bool Result, const llvm::Twine &Reason, T &Element) const {
    if (not Result) {
      std::string Buffer;
      {
        llvm::raw_string_ostream StringStream(Buffer);
        StringStream << Reason << "\n";
        serialize(StringStream, const_cast<std::remove_const_t<T> &>(Element));
      }

      if (AssertOnFail) {
        revng_abort(Buffer.c_str());
      } else {
        revng_log(ModelVerifyLogger, Buffer);
      }
    }

    return Result;
  }

  bool fail() const { return maybeFail(false); }
  bool fail(const llvm::Twine &Reason) const {
    return maybeFail(false, Reason);
  }

  template<typename T>
  bool fail(const llvm::Twine &Reason, T &Element) const {
    return maybeFail(false, Reason, Element);
  }
};

} // namespace model
