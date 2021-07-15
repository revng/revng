#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <map>
#include <set>

#include "revng/Support/Assert.h"

namespace model {
class Type;
} // namespace model
namespace model {

class VerifyHelper {
private:
  std::set<const model::Type *> VerifiedCache;
  std::map<const model::Type *, uint64_t> SizeCache;
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
  bool isVerificationInProgess(const model::Type *T) const {
    return InProgress.count(T) != 0;
  }

  void verificationInProgess(const model::Type *T) {
    revng_assert(not isVerificationInProgess(T));
    revng_assert(not isVerified(T));
    InProgress.insert(T);
  }

  void verificationCompleted(const model::Type *T) {
    revng_assert(isVerificationInProgess(T));
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
      return {};
  }

public:
  bool maybeFail(bool Result) const {
    if (AssertOnFail and not Result) {
      revng_abort();
    } else {
      return Result;
    }
  }

  bool fail() const { return maybeFail(false); }
};

} // namespace model
