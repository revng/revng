#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/IR/Module.h"

#include "revng/Support/ManagedStaticRegistry.h"

struct IRHelper {
public:
  std::string Name;
  std::string Description;

  // TODO: pull the prototype inside

public:
  const std::string &key() const { return Name; }
};
using RegisterIRHelper = RegisterManagedStaticImpl<IRHelper>;

namespace detail {
inline void assertIRHelperWasRegistered(llvm::StringRef Name) {
  if (not RegisterIRHelper::get(Name)) {
    std::string Error = "`" + Name.str()
                        + "` is not a known helper.\n"
                          "Did you forget to register it?";
    revng_abort(Error.c_str());
  }
}
} // namespace detail

inline const llvm::Function *getIRHelper(llvm::StringRef Name,
                                         const llvm::Module &Module) {
  detail::assertIRHelperWasRegistered(Name);

  return Module.getFunction(Name);
}

inline llvm::Function *getIRHelper(llvm::StringRef Name, llvm::Module &Module) {
  detail::assertIRHelperWasRegistered(Name);

  return Module.getFunction(Name);
}

inline llvm::FunctionCallee
getOrInsertIRHelper(llvm::StringRef Name,
                    llvm::Module &Module,
                    llvm::FunctionType *FunctionType) {
  detail::assertIRHelperWasRegistered(Name);

  return Module.getOrInsertFunction(Name, FunctionType);
}

inline llvm::FunctionCallee getOrInsertIRHelper(llvm::StringRef Name,
                                                llvm::Module &Module,
                                                llvm::Type *ReturnValueType) {
  detail::assertIRHelperWasRegistered(Name);

  return Module.getOrInsertFunction(Name, ReturnValueType);
}

inline llvm::Function *createIRHelper(llvm::StringRef Name,
                                      llvm::Module &Module,
                                      llvm::FunctionType *FunctionType,
                                      llvm::GlobalValue::LinkageTypes Linkage) {
  detail::assertIRHelperWasRegistered(Name);

  return llvm::Function::Create(FunctionType, Linkage, Name, Module);
}
