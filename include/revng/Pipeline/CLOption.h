#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"

#include "revng/Support/Assert.h"

namespace pipeline {

class CLOptionBase {
public:
  explicit CLOptionBase(llvm::StringRef Name) {
    getRegisteredOptions()[Name] = this;
  }
  virtual ~CLOptionBase() = default;
  virtual bool isSet() const = 0;
  virtual std::string get() const = 0;
  virtual llvm::StringRef name() const = 0;

  static const CLOptionBase &getOption(llvm::StringRef Name) {
    std::string Msg = "option has not been registered " + Name.str();

    revng_assert(hasOption(Name), Msg.c_str());
    return *getRegisteredOptions()[Name];
  }

  static bool hasOption(llvm::StringRef Name) {
    return getRegisteredOptions().count(Name) != 0;
  }

private:
  static llvm::StringMap<const CLOptionBase *> &getRegisteredOptions() {
    static llvm::StringMap<const CLOptionBase *> RegisteredOption;
    return RegisteredOption;
  }
};

template<typename T>
class CLOptionWrapper : public CLOptionBase {
public:
  CLOptionWrapper(const CLOptionWrapper &) = delete;
  CLOptionWrapper(CLOptionWrapper &&) = delete;
  CLOptionWrapper &operator=(const CLOptionWrapper &) = delete;
  CLOptionWrapper &operator=(CLOptionWrapper &&) = delete;

  template<typename... Args>
  CLOptionWrapper(llvm::StringRef InvokableType,
                  llvm::StringRef Name,
                  Args &&...Arguments) :
    CLOptionBase(Name),
    FullName((InvokableType + "-" + Name).str()),
    Option(llvm::StringRef(FullName), std::forward<Args>(Arguments)...) {}

  bool isSet() const override { return Option.getNumOccurrences() != 0; }
  std::string get() const override {
    std::string ToReturn;
    llvm::raw_string_ostream SO(ToReturn);
    SO << Option.getValue();
    SO.flush();
    return ToReturn;
  }
  llvm::StringRef name() const override { return Option.ArgStr; }

private:
  std::string FullName;
  llvm::cl::opt<T> Option;
};
} // namespace pipeline
