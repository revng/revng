//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/IR/IRBuilder.h"
#include "llvm/Pass.h"

using namespace llvm;

// This pass is needed by the llvm-to-clift conversion, because - unlike LLVM IR
// - Clift does not permit array prvalues. This makes the conversion much more
// difficult in those cases. For that reason any array constant returns are
// converted into allocas of the same array type followed by writes for each
// aggregate element initialiser found in the array constant.
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
      // First, detect any basic block with a llvm::ReturnInst terminator whose
      // return value is an array constant.

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

      // Once an array constant return is identified, create an alloca of the
      // same type and initialise it using stores for each defined aggregate
      // initialiser found in the array constant.

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

      // Finally, replace the return by loading the array value through the
      // alloca pointer. This corresponds to an lvalue expression instead of a
      // prvalue expression in the conversion to Clift.

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
