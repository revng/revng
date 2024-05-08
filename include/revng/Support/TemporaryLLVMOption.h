#pragma once

//
// This file is distributed under the MIT License. See LICENSE.mit for details.
//

#include "llvm/Support/CommandLine.h"

#include "revng/Support/IRHelpers.h"

/// Temporarily change the value of a llvm::cl option
template<typename T>
struct TemporaryLLVMOption {
public:
  TemporaryLLVMOption(const char *Name, const T &Value) :
    Name(Name), Options(llvm::cl::getRegisteredOptions()) {
    OldValue = Opt(Options, Name)->getValue();
    Opt(Options, Name)->setInitialValue(Value);
  }

  ~TemporaryLLVMOption() { Opt(Options, Name)->setInitialValue(OldValue); }

private:
  T OldValue;
  const char *Name;
  llvm::StringMap<llvm::cl::Option *> &Options;
  static constexpr const auto &Opt = getOption<T>;
};
