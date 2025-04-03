//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/IR/IRBuilder.h"
#include "llvm/Pass.h"

using namespace llvm;

struct RemoveConstantArrayReturns : public FunctionPass {
public:
  static char ID;

  RemoveConstantArrayReturns() : FunctionPass(ID) {}

  bool runOnFunction(Function &F) override {
    auto *PtrT = PointerType::get(F.getContext(), /*AddrSpace=*/0);
    auto *IntPtrT = F.getParent()
                      ->getDataLayout()
                      .getIntPtrType(F.getContext(), /*AddrSpace=*/0);

    bool Changed = false;
    for (BasicBlock &BB : F) {
      auto *Return = dyn_cast<ReturnInst>(BB.getTerminator());
      if (Return == nullptr)
        continue;

      auto *Array = dyn_cast_or_null<ConstantArray>(Return->getReturnValue());
      if (Array == nullptr)
        continue;

      auto *ArrayT = cast<ArrayType>(Array->getType());
      auto *ElementT = ArrayT->getElementType();

      if (auto *IntegerT = dyn_cast<IntegerType>(ElementT);
          IntegerT == nullptr or IntegerT->getBitWidth() != 8)
        continue;

      IRBuilder<> Builder(&BB, BasicBlock::iterator(Return));
      auto *Alloca = Builder.CreateAlloca(ArrayT);
      auto *AllocaInt = Builder.CreatePtrToInt(Alloca, IntPtrT);

      for (unsigned I = 0, C = ArrayT->getNumElements(); I < C; ++I) {
        auto *Initializer = Array->getAggregateElement(I);
        if (not isa<UndefValue>(Initializer)) {
          auto *Offset = Builder.getIntN(IntPtrT->getBitWidth(), I);
          auto *ElementInt = Builder.CreateAdd(AllocaInt, Offset);
          auto *Element = Builder.CreateIntToPtr(ElementInt, PtrT);
          Builder.CreateStore(Initializer, Element);
        }
      }

      Builder.CreateRet(Builder.CreateLoad(ArrayT, Alloca));
      Return->eraseFromParent();
      Changed = true;
    }
    return Changed;
  }
};

char RemoveConstantArrayReturns::ID = 0;

using Register = RegisterPass<RemoveConstantArrayReturns>;
static Register Y("remove-constant-array-returns",
                  "RemoveConstantArrayReturns",
                  false,
                  false);
