#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"

#include "revng/ADT/Concepts.h"
#include "revng/Support/Assert.h"

namespace pipeline {

template<typename T>
concept PipelineOptionType = anyOf<T, int, uint64_t, std::string>();

template<PipelineOptionType T>
class CLOptionWrapper;

class CLOptionBase {
private:
  const char *ID;

public:
  explicit CLOptionBase(const char *ID, llvm::StringRef Name) : ID(ID) {
    auto &Map = getRegisteredOptions();
    revng_assert(Map.count(Name) == 0);
    Map[Name] = this;
  }

  virtual ~CLOptionBase() = default;

  const char *getID() const { return ID; }

  template<PipelineOptionType T>
  static const CLOptionWrapper<T> &getOption(llvm::StringRef Name) {
    auto &Map = getRegisteredOptions();

    std::string Msg = "option has not been registered " + Name.str();
    revng_assert(Map.count(Name) != 0, Msg.c_str());

    return *llvm::dyn_cast<const CLOptionWrapper<T>>(Map[Name]);
  }

private:
  static llvm::StringMap<const CLOptionBase *> &getRegisteredOptions() {
    static llvm::StringMap<const CLOptionBase *> RegisteredOption;
    return RegisteredOption;
  }
};

template<PipelineOptionType T>
class CLOptionWrapper final : public CLOptionBase {
private:
  std::string FullName;
  llvm::cl::opt<T> Option;

  static const char &getID() {
    static char ID;
    return ID;
  }

public:
  CLOptionWrapper(const CLOptionWrapper &) = delete;
  CLOptionWrapper(CLOptionWrapper &&) = delete;
  CLOptionWrapper &operator=(const CLOptionWrapper &) = delete;
  CLOptionWrapper &operator=(CLOptionWrapper &&) = delete;

  template<typename... Args>
  CLOptionWrapper(llvm::StringRef InvokableType,
                  llvm::StringRef Name,
                  Args &&...Arguments) :
    CLOptionBase(&getID(), (InvokableType + "-" + Name).str()),
    FullName((InvokableType + "-" + Name).str()),
    Option(llvm::StringRef(FullName), std::forward<Args>(Arguments)...) {}

  static bool classof(const CLOptionBase *Base) {
    return Base->getID() == &getID();
  }

  bool isSet() const { return Option.getNumOccurrences() != 0; }
  llvm::StringRef name() const { return Option.ArgStr; }
  T get() const { return Option.getValue(); }
};

} // namespace pipeline
