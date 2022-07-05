#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <optional>

#include "llvm/Support/Error.h"

#include "revng/Support/Assert.h"

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

  llvm::Error *get(size_t Index) { return &Errors.at(Index); }

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
};

/// Wrapper class to handle nullptr
/// PipelineC can accept a nullptr as an ErrorList argument, this class wraps
/// the pointer so that if it is nullptr it will point to the throwaway local
/// ErrorList EL
class ErrorListWrapper {
private:
  ErrorList EL;
  ErrorList *ELP;

public:
  ErrorListWrapper(ErrorList *EL) : EL() {
    if (EL == nullptr) {
      ELP = &this->EL;
    } else {
      ELP = EL;
    }
  }

  ErrorList &operator*() { return *ELP; }
  ErrorList *operator->() { return ELP; }
};
