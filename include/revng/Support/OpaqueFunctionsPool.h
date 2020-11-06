#ifndef OPAQUEFUNCTIONSPOOL_H
#define OPAQUEFUNCTIONSPOOL_H

// Standard includes
#include <map>

// LLVM includes
#include "llvm/ADT/Twine.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/LLVMContext.h"

// Local libraries includes
#include "revng/Support/Assert.h"

template<typename KeyT>
class OpaqueFunctionsPool {
private:
  llvm::Module *M;
  llvm::LLVMContext &Context;
  const bool PurgeOnDestruction;
  std::map<KeyT, llvm::Function *> Pool;
  llvm::AttributeList AttributeSets;

public:
  OpaqueFunctionsPool(llvm::Module *M, bool PurgeOnDestruction) :
    M(M), Context(M->getContext()), PurgeOnDestruction(PurgeOnDestruction) {}

  ~OpaqueFunctionsPool() {
    if (PurgeOnDestruction) {
      for (auto &[Key, F] : Pool) {
        revng_assert(F->use_begin() == F->use_end());
        F->eraseFromParent();
      }
    }
  }

public:
  void addFnAttribute(llvm::Attribute::AttrKind Kind) {
    using namespace llvm;
    AttributeSets = AttributeSets.addAttribute(Context,
                                               AttributeList::FunctionIndex,
                                               Kind);
  }

public:
  llvm::Function *
  get(KeyT Key, llvm::FunctionType *FT, const llvm::Twine &Name = {}) {
    using namespace llvm;

    Function *F = nullptr;
    auto It = Pool.find(Key);
    if (It != Pool.end()) {
      F = It->second;
    } else {
      F = Function::Create(FT, GlobalValue::ExternalLinkage, Name, M);
      F->setAttributes(AttributeSets);
      Pool.insert(It, { Key, F });
    }

    // Ensure the function we're returning is as expected
    revng_assert(F->getType()->getPointerElementType() == FT);

    return F;
  }

  llvm::Function *get(KeyT Key,
                      llvm::Type *ReturnType = nullptr,
                      llvm::ArrayRef<llvm::Type *> Arguments = {},
                      const llvm::Twine &Name = {}) {
    using namespace llvm;
    if (ReturnType == nullptr)
      ReturnType = Type::getVoidTy(Context);

    return get(Key, FunctionType::get(ReturnType, Arguments, false), Name);
  }
};

#endif // OPAQUEFUNCTIONSPOOL_H
