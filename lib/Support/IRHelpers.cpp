/// \file IRHelpers.cpp
/// \brief Implementation of IR helper functions

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <fstream>

#include "llvm/Support/raw_os_ostream.h"

#include "revng/BasicAnalyses/GeneratedCodeBasicInfo.h"
#include "revng/Support/IRHelpers.h"

// TODO: including GeneratedCodeBasicInfo.h is not very nice

using namespace llvm;

void dumpModule(const Module *M, const char *Path) {
  std::ofstream FileStream(Path);
  raw_os_ostream Stream(FileStream);
  M->print(Stream, nullptr, true);
}

PointerType *getStringPtrType(LLVMContext &C) {
  return Type::getInt8Ty(C)->getPointerTo();
}

GlobalVariable *buildString(Module *M, StringRef String, const Twine &Name) {
  LLVMContext &C = M->getContext();
  auto *Initializer = ConstantDataArray::getString(C, String, true);
  return new GlobalVariable(*M,
                            Initializer->getType(),
                            true,
                            GlobalVariable::InternalLinkage,
                            Initializer,
                            Name);
}

Constant *buildStringPtr(Module *M, StringRef String, const Twine &Name) {
  LLVMContext &C = M->getContext();
  GlobalVariable *NewVariable = buildString(M, String, Name);
  return ConstantExpr::getBitCast(NewVariable, getStringPtrType(C));
}

StringRef extractFromConstantStringPtr(Value *V) {
  auto *ConstantGEP = dyn_cast<ConstantExpr>(V);
  if (ConstantGEP == nullptr)
    return {};

  auto *NoCasts = ConstantGEP->stripPointerCasts();
  auto *GV = dyn_cast_or_null<GlobalVariable>(NoCasts);
  if (GV == nullptr)
    return {};

  auto *Initializer = dyn_cast_or_null<ConstantDataArray>(GV->getInitializer());
  if (Initializer == nullptr or not Initializer->isCString())
    return {};

  return Initializer->getAsCString();
}

Constant *getUniqueString(Module *M,
                          StringRef Namespace,
                          StringRef String,
                          const Twine &Name) {
  LLVMContext &C = M->getContext();
  NamedMDNode *StringsList = M->getOrInsertNamedMetadata(Namespace);
  auto *Int8PtrTy = getStringPtrType(C);

  for (MDNode *Operand : StringsList->operands()) {
    auto *T = cast<MDTuple>(Operand);
    revng_assert(T->getNumOperands() == 1);
    auto *CAM = cast<ConstantAsMetadata>(T->getOperand(0).get());
    auto *GV = cast<GlobalVariable>(CAM->getValue());
    revng_assert(GV->isConstant() and GV->hasInitializer());

    const Constant *Initializer = GV->getInitializer();
    StringRef Content = cast<ConstantDataArray>(Initializer)->getAsString();

    // Ignore the terminator
    if (Content.drop_back() == String)
      return ConstantExpr::getBitCast(GV, Int8PtrTy);
  }

  GlobalVariable *NewVariable = buildString(M, String, Name);
  auto *CAM = ConstantAsMetadata::get(NewVariable);
  StringsList->addOperand(MDTuple::get(C, { CAM }));
  return ConstantExpr::getBitCast(NewVariable, Int8PtrTy);
}

CallInst *getLastNewPC(Instruction *TheInstruction) {
  std::set<BasicBlock *> Visited;
  std::queue<BasicBlock::reverse_iterator> WorkList;

  // Initialize WorkList with an iterator pointing at the given instruction
  if (TheInstruction->getIterator() == TheInstruction->getParent()->begin())
    WorkList.push(--TheInstruction->getParent()->rend());
  else
    WorkList.push(++TheInstruction->getReverseIterator());

  // Process the worklist
  while (not WorkList.empty()) {
    auto I = WorkList.front();
    WorkList.pop();
    auto *BB = I->getParent();
    auto End = BB->rend();

    // Go through the instructions looking for calls to newpc
    for (; I != End; I++)
      if (CallInst *Marker = getCallTo(&*I, "newpc"))
        return Marker;

    // If we didn't find a newpc call yet, continue exploration backward
    // If one of the predecessors is the dispatcher, don't explore any further
    for (BasicBlock *Predecessor : predecessors(BB)) {
      using GCBI = GeneratedCodeBasicInfo;

      // Assert we didn't reach the almighty dispatcher
      revng_assert(GCBI::isPartOfRootDispatcher(Predecessor) == false);

      // Ignore already visited or empty BBs
      if (!Predecessor->empty() && Visited.find(Predecessor) == Visited.end()) {
        WorkList.push(Predecessor->rbegin());
        Visited.insert(Predecessor);
      }
    }
  }

  return nullptr;
}

std::pair<MetaAddress, uint64_t> getPC(Instruction *TheInstruction) {
  CallInst *NewPCCall = getLastNewPC(TheInstruction);

  // Couldn't find the current PC
  if (NewPCCall == nullptr)
    return { MetaAddress::invalid(), 0 };

  auto PC = MetaAddress::fromConstant(NewPCCall->getArgOperand(0));
  uint64_t Size = getLimitedValue(NewPCCall->getArgOperand(1));
  revng_assert(Size != 0);
  return { PC, Size };
}

Function *changeFunctionType(Function &OldFunction,
                             Type *NewReturnType,
                             ArrayRef<Type *> NewArguments) {
  //
  // Validation
  //
  FunctionType &OldFunctionType = *OldFunction.getFunctionType();

  // Either the old type was returning void or the return type has to be same
  auto OldReturnType = OldFunctionType.getReturnType();
  if (NewReturnType != nullptr) {
    if (not OldReturnType->isVoidTy())
      revng_assert(OldReturnType == NewReturnType);
  } else {
    NewReturnType = OldReturnType;
  }

  // New arguments
  SmallVector<Type *> NewFunctionArguments;
  llvm::copy(OldFunctionType.params(),
             std::back_inserter(NewFunctionArguments));
  llvm::copy(NewArguments, std::back_inserter(NewFunctionArguments));

  auto &NewFunctionType = *FunctionType::get(NewReturnType,
                                             NewFunctionArguments,
                                             OldFunctionType.isVarArg());

  //
  // Recreate the function as similar as possible
  //
  auto *NewFunction = Function::Create(&NewFunctionType,
                                       GlobalValue::ExternalLinkage,
                                       "",
                                       OldFunction.getParent());
  NewFunction->takeName(&OldFunction);
  NewFunction->copyAttributesFrom(&OldFunction);
  NewFunction->copyMetadata(&OldFunction, 0);

  // Steal body
  std::vector<BasicBlock *> Body;
  for (BasicBlock &BB : OldFunction)
    Body.push_back(&BB);
  auto &NewBody = NewFunction->getBasicBlockList();
  for (BasicBlock *BB : Body) {
    BB->removeFromParent();
    revng_assert(BB->getParent() == nullptr);
    NewBody.push_back(BB);
    revng_assert(BB->getParent() == NewFunction);
  }

  // Replace arguments and copy their names
  unsigned I = 0;
  for (Argument &OldArgument : OldFunction.args()) {
    Argument &NewArgument = *NewFunction->getArg(I);
    NewArgument.setName(OldArgument.getName());
    OldArgument.replaceAllUsesWith(&NewArgument);
    ++I;
  }

  // We do not delete OldFunction in order not to break call sites

  return NewFunction;
}

void dumpUsers(llvm::Value *V) {
  using namespace llvm;

  struct InstructionUser {
    Function *F;
    BasicBlock *BB;
    Instruction *I;
    bool operator<(const InstructionUser &Other) const {
      return std::tie(F, BB, I) < std::tie(Other.F, Other.BB, Other.I);
    }
  };
  SmallVector<InstructionUser> InstructionUsers;
  for (User *U : V->users()) {
    if (auto *I = dyn_cast<Instruction>(U)) {
      BasicBlock *BB = I->getParent();
      Function *F = BB->getParent();
      InstructionUsers.push_back({ F, BB, I });
    } else {
      dbg << "  ";
      U->dump();
    }
  }

  llvm::sort(InstructionUsers);

  Function *LastF = nullptr;
  BasicBlock *LastBB = nullptr;
  for (InstructionUser &IU : InstructionUsers) {
    if (IU.F != LastF) {
      LastF = IU.F;
      dbg << "  Function " << getName(LastF) << "\n";
    }

    if (IU.BB != LastBB) {
      LastBB = IU.BB;
      dbg << "    Block " << getName(LastBB) << "\n";
    }

    dbg << "    ";
    IU.I->dump();
  }
}
