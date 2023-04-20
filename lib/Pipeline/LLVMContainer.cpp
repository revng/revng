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
#include "revng/Pipeline/LLVMGlobalKindBase.h"

const char pipeline::LLVMContainer::ID = '0';
using namespace pipeline;

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

std::unique_ptr<ContainerBase>
LLVMContainer::cloneFiltered(const TargetsList &Targets) const {
  using InspectorT = LLVMGlobalKindBase;
  auto ToClone = InspectorT::functions(Targets, *this->self());
  auto ToClonedNotOwned = InspectorT::untrackedFunctions(*this->self());

  const auto Filter = [&ToClone, &ToClonedNotOwned](const auto &GlobalSym) {
    if (not llvm::isa<llvm::Function>(GlobalSym))
      return true;

    const auto &F = llvm::cast<llvm::Function>(GlobalSym);
    return ToClone.count(F) != 0 or ToClonedNotOwned.count(F) != 0;
  };

  llvm::ValueToValueMapTy Map;

  revng::verify(Module.get());
  auto Cloned = llvm::CloneModule(*Module, Map, Filter);

  for (auto &Function : Module->functions()) {
    auto *Other = Cloned->getFunction(Function.getName());
    if (not Other)
      continue;

    Other->clearMetadata();
    llvm::SmallVector<std::pair<unsigned, llvm::MDNode *>, 2> MDs;
    Function.getAllMetadata(MDs);
    for (auto &MD : MDs) {
      // The !dbg attachment from the function defintion cannot be attached to
      // its declaration.
      if (Other->isDeclaration() && isa<llvm::DISubprogram>(MD.second))
        continue;

      Other->addMetadata(MD.first, *llvm::MapMetadata(MD.second, Map));
    }
  }

  return std::make_unique<ThisType>(this->name(), this->Ctx, std::move(Cloned));
}
