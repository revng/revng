#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <optional>

#include "llvm/Support/Error.h"

#include "revng/Support/Assert.h"
#include "revng/Support/Debug.h"

class ErrorList {
private:
  std::vector<llvm::Error> Errors;

public:
  ErrorList() = default;
  ErrorList(ErrorList &&EC) = default;
  ErrorList &operator=(ErrorList &&) = default;
  ErrorList(const ErrorList &EC) = delete;
  ErrorList &operator=(const ErrorList &) = delete;

  ErrorList(llvm::Error &&Error) : Errors() { push_back(std::move(Error)); }

  ~ErrorList() {
    for (auto &Error : Errors) {
      llvm::consumeError(std::move(Error));
    }
  }

  operator bool() const { return !empty(); }

  bool empty() const { return Errors.empty(); }

  void push_back(llvm::Error &&Error) {
    if (Error)
      Errors.push_back(std::move(Error));
    else
      llvm::consumeError(std::move(Error));
  }

  void merge(ErrorList &&EL) {
    for (auto &Error : EL.Errors) {
      push_back(std::move(Error));
    }
    EL.Errors.clear();
  }

  size_t size() const { return Errors.size(); }

  const llvm::Error *get(size_t Index) const { return &Errors.at(Index); }

  template<typename T>
  T fail(llvm::Error &&Error, T ReturnValue) {
    push_back(std::move(Error));
    return ReturnValue;
  }

  template<typename T>
  T failIf(llvm::Error &&Error, T SuccessValue, T FailValue) {
    if (Error) {
      push_back(std::move(Error));
      return FailValue;
    }
    return SuccessValue;
  }

  bool failIf(llvm::Error &&Error) {
    return failIf(std::move(Error), true, false);
  }

  std::string serialize() const {
    std::string Out;
    llvm::raw_string_ostream RSS(Out);
    for (const llvm::Error &Error : Errors)
      RSS << Error << '\n';
    RSS.flush();
    return Out;
  }

  void dump(llvm::raw_ostream &OS) const { OS << serialize(); }

  void dump(std::ostream &OS) const { OS << serialize(); }

  void dump() const debug_function { dump(dbg); }

  friend llvm::raw_ostream &
  operator<<(llvm::raw_ostream &OS, const ErrorList &EL) {
    EL.dump(OS);
    return OS;
  }

  friend std::ostream &operator<<(std::ostream &OS, const ErrorList &EL) {
    EL.dump(OS);
    return OS;
  }
};
