/// \file LLVMContainer.cpp
/// \brief A llvm container is a container which uses a llvm module as a
/// backend, and can be customized with downstream kinds that specify which
/// global objects in it are which target.

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <memory>

#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Constant.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"

#include "revng/Pipeline/LLVMContainer.h"

const char pipeline::LLVMContainer::ID = '0';

void pipeline::makeGlobalObjectsArray(llvm::Module &Module,
                                      llvm::StringRef GlobalArrayName) {
  auto *IntegerTy = llvm::IntegerType::get(Module.getContext(),
                                           Module.getDataLayout()
                                             .getPointerSizeInBits());

  llvm::SmallVector<llvm::Constant *, 10> Globals;

  for (auto &Global : Module.globals())
    Globals.push_back(llvm::ConstantExpr::getPtrToInt(&Global, IntegerTy));

  for (auto &Global : Module.functions())
    if (not Global.isIntrinsic())
      Globals.push_back(llvm::ConstantExpr::getPtrToInt(&Global, IntegerTy));

  auto *GlobalArrayType = llvm::ArrayType::get(IntegerTy, Globals.size());

  auto *Initilizer = llvm::ConstantArray::get(GlobalArrayType, Globals);

  new llvm::GlobalVariable(Module,
                           GlobalArrayType,
                           false,
                           llvm::GlobalValue::LinkageTypes::ExternalLinkage,
                           Initilizer,
                           GlobalArrayName);
}
