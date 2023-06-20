/// \file IRHelpers.cpp
/// Implementation of IR helper functions.

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <fstream>

#include "llvm/ADT/SmallSet.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/TypedPointerType.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/SHA1.h"
#include "llvm/Support/raw_os_ostream.h"

#include "revng/ADT/Queue.h"
#include "revng/ADT/RecursiveCoroutine.h"
#include "revng/Support/BlockType.h"
#include "revng/Support/IRHelpers.h"

// TODO: including GeneratedCodeBasicInfo.h is not very nice

using namespace llvm;

void dumpModule(const Module *M, const char *Path) {
  std::ofstream FileStream(Path);
  raw_os_ostream Stream(FileStream);
  M->print(Stream, nullptr, false);
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

StringRef extractFromConstantStringPtr(Value *V) {
  revng_assert(V->getType()->isPointerTy());

  auto *GV = dyn_cast_or_null<GlobalVariable>(V);
  if (GV == nullptr)
    return {};

  auto *Initializer = dyn_cast_or_null<ConstantDataArray>(GV->getInitializer());
  if (Initializer == nullptr or not Initializer->isCString())
    return {};

  return Initializer->getAsCString();
}

static std::string mangleName(StringRef String) {
  auto IsPrintable = [](StringRef String) { return all_of(String, isPrint); };

  auto ContainsSpaces = [](StringRef String) {
    return any_of(String, isSpace);
  };

  constexpr auto SHA1HexLength = 40;
  if (String.size() > SHA1HexLength or not IsPrintable(String)
      or ContainsSpaces(String)) {
    ArrayRef Data(reinterpret_cast<const uint8_t *>(String.data()),
                  String.size());
    return llvm::toHex(SHA1::hash(Data), true);
  } else {
    return String.str();
  }
}

Constant *getUniqueString(Module *M, StringRef String, StringRef Namespace) {
  revng_assert(not Namespace.empty());

  LLVMContext &Context = M->getContext();
  std::string GlobalName = (Twine(Namespace) + mangleName(String)).str();
  auto *Global = M->getGlobalVariable(GlobalName);

  if (Global != nullptr) {
    revng_assert(Global->hasInitializer());
    if (not String.empty()) {
      auto Initializer = cast<ConstantDataSequential>(Global->getInitializer());
      revng_assert(Initializer->isCString());
      revng_assert(Initializer->getAsCString() == String);
    } else {
      revng_assert(isa<ConstantAggregateZero>(Global->getInitializer()));
    }
  } else {
    // This may return a ConstantAggregateZero in case of empty String.
    Constant *Initializer = ConstantDataArray::getString(Context,
                                                         String,
                                                         /* AddNull */ true);
    revng_assert(isa<ConstantDataArray>(Initializer)
                 or isa<ConstantAggregateZero>(Initializer));
    if (String.empty()) {
      revng_assert(isa<ConstantAggregateZero>(Initializer));
    } else {
      auto CDAInitializer = cast<ConstantDataArray>(Initializer);
      revng_assert(CDAInitializer->isCString());
      revng_assert(CDAInitializer->getAsCString() == String);
    }

    Global = new GlobalVariable(*M,
                                Initializer->getType(),
                                /* isConstant */ true,
                                GlobalValue::LinkOnceODRLinkage,
                                Initializer,
                                GlobalName);
  }

  auto *Int8PtrTy = getStringPtrType(Context);
  return ConstantExpr::getBitCast(Global, Int8PtrTy);
}

CallInst *getLastNewPC(Instruction *TheInstruction) {
  CallInst *Result = nullptr;
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
    bool Stop = false;
    for (; not Stop and I != End; I++) {
      if (CallInst *Marker = getCallTo(&*I, "newpc")) {
        if (Result != nullptr)
          return nullptr;
        Result = Marker;
        Stop = true;
      }
    }

    if (Stop)
      continue;

    // If we didn't find a newpc call yet, continue exploration backward
    // If one of the predecessors is the dispatcher, don't explore any further
    for (BasicBlock *Predecessor : predecessors(BB)) {
      // Assert we didn't reach the almighty dispatcher
      revng_assert(isPartOfRootDispatcher(Predecessor) == false);

      // Ignore already visited or empty BBs
      if (!Predecessor->empty() && !Visited.contains(Predecessor)) {
        WorkList.push(Predecessor->rbegin());
        Visited.insert(Predecessor);
      }
    }
  }

  return Result;
}

std::pair<MetaAddress, uint64_t> getPC(Instruction *TheInstruction) {
  CallInst *NewPCCall = getLastNewPC(TheInstruction);

  // Couldn't find the current PC
  if (NewPCCall == nullptr)
    return { MetaAddress::invalid(), 0 };

  MetaAddress PC = blockIDFromNewPC(NewPCCall).start();
  using namespace NewPCArguments;
  uint64_t Size = getLimitedValue(NewPCCall->getArgOperand(InstructionSize));
  revng_assert(Size != 0);
  return { PC, Size };
}

/// Boring code to get the text of the metadata with the specified kind
/// associated to the given instruction
StringRef getText(const Instruction *I, unsigned Kind) {
  revng_assert(I != nullptr);

  Metadata *MD = I->getMetadata(Kind);

  if (MD == nullptr)
    return StringRef();

  auto Node = dyn_cast<MDNode>(MD);

  revng_assert(Node != nullptr);

  const MDOperand &Operand = Node->getOperand(0);

  Metadata *MDOperand = Operand.get();

  if (MDOperand == nullptr)
    return StringRef();

  if (auto *String = dyn_cast<MDString>(MDOperand)) {
    return String->getString();
  } else if (auto *CAM = dyn_cast<ConstantAsMetadata>(MDOperand)) {
    auto *Cast = cast<ConstantExpr>(CAM->getValue());
    auto *GV = cast<GlobalVariable>(Cast->getOperand(0));
    auto *Initializer = GV->getInitializer();
    return cast<ConstantDataArray>(Initializer)->getAsString().drop_back();
  } else {
    revng_abort();
  }
}

void moveBlocksInto(Function &OldFunction, Function &NewFunction) {
  // Steal body
  std::vector<BasicBlock *> Body;
  for (BasicBlock &BB : OldFunction)
    Body.push_back(&BB);
  for (BasicBlock *BB : Body) {
    BB->removeFromParent();
    revng_assert(BB->getParent() == nullptr);
    NewFunction.insert(NewFunction.end(), BB);
    revng_assert(BB->getParent() == &NewFunction);
  }
}

Function &moveToNewFunctionType(Function &OldFunction, FunctionType &NewType) {
  //
  // Recreate the function as similar as possible
  //
  auto *NewFunction = Function::Create(&NewType,
                                       GlobalValue::ExternalLinkage,
                                       "",
                                       OldFunction.getParent());
  NewFunction->takeName(&OldFunction);
  NewFunction->copyAttributesFrom(&OldFunction);
  NewFunction->copyMetadata(&OldFunction, 0);

  // Steal body
  if (not OldFunction.isDeclaration())
    moveBlocksInto(OldFunction, *NewFunction);

  return *NewFunction;
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

  Function &NewFunction = moveToNewFunctionType(OldFunction, NewFunctionType);

  // Replace arguments and copy their names
  unsigned I = 0;
  for (Argument &OldArgument : OldFunction.args()) {
    Argument &NewArgument = *NewFunction.getArg(I);
    NewArgument.setName(OldArgument.getName());
    OldArgument.replaceAllUsesWith(&NewArgument);
    ++I;
  }

  // We do not delete OldFunction in order not to break call sites

  return &NewFunction;
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

static RecursiveCoroutine<void>
findJumpTarget(const llvm::BasicBlock *&Result,
               const llvm::BasicBlock *BB,
               std::set<const BasicBlock *> &Visited) {
  Visited.insert(BB);

  if (isJumpTarget(BB)) {
    revng_assert(Result == nullptr,
                 "This block leads to multiple jump targets");
    Result = BB;
  } else {
    for (const BasicBlock *Predecessor : predecessors(BB)) {
      if (!Visited.contains(Predecessor))
        rc_recur findJumpTarget(Result, Predecessor, Visited);
    }
  }

  rc_return;
}

const llvm::BasicBlock *getJumpTargetBlock(const llvm::BasicBlock *BB) {
  const llvm::BasicBlock *Result = nullptr;
  std::set<const BasicBlock *> Visited;
  findJumpTarget(Result, BB, Visited);
  return Result;
}

void pruneDICompileUnits(Module &M) {
  auto *CUs = M.getNamedMetadata("llvm.dbg.cu");
  if (CUs == nullptr)
    return;

  // Purge CUs list
  CUs->clearOperands();

  std::set<DICompileUnit *> Reachable;
  DebugInfoFinder DIFinder;
  DIFinder.processModule(M);

  for (DICompileUnit *CU : DIFinder.compile_units())
    Reachable.insert(CU);

  if (Reachable.size() == 0) {
    CUs->eraseFromParent();
  } else {
    // Recreate CUs list
    for (DICompileUnit *CU : Reachable)
      CUs->addOperand(CU);
  }
}

using ValueSet = SmallSet<Value *, 2>;

static RecursiveCoroutine<void>
findPhiTreeLeavesImpl(ValueSet &Leaves, ValueSet &Visited, llvm::Value *V) {
  if (auto *Phi = dyn_cast<PHINode>(V)) {
    revng_assert(!Visited.contains(V));
    Visited.insert(V);
    for (Value *Operand : Phi->operands())
      rc_recur findPhiTreeLeavesImpl(Leaves, Visited, Operand);
  } else {
    Leaves.insert(V);
  }

  rc_return;
}

ValueSet findPhiTreeLeaves(Value *Root) {
  ValueSet Result;
  ValueSet Visited;
  findPhiTreeLeavesImpl(Result, Visited, Root);
  return Result;
}
void revng::verify(const llvm::Module *M) {
  if (VerifyLog.isEnabled())
    forceVerify(M);
}

void revng::verify(const llvm::Function *F) {
  if (VerifyLog.isEnabled())
    forceVerify(F);
}

void revng::forceVerify(const llvm::Module *M) {
  // NOLINTNEXTLINE
  if (llvm::verifyModule(*M, &llvm::dbgs()) != 0) {
    int FD = 0;
    SmallString<128> Path;
    auto EC = llvm::sys::fs::createTemporaryFile("revng-failed-verify",
                                                 "ll",
                                                 FD,
                                                 Path);
    revng_assert(!EC and FD != 0);
    llvm::raw_fd_ostream Stream(FD, true);
    M->print(Stream, nullptr);
    dbg << "Module printed to " << Path.str().str() << "\n";
    revng_abort();
  }
}

void revng::forceVerify(const llvm::Function *F) {
  // NOLINTNEXTLINE
  if (llvm::verifyFunction(*F, &llvm::dbgs()) != 0) {
    int FD = 0;
    SmallString<128> Path;
    auto EC = llvm::sys::fs::createTemporaryFile("revng-failed-verify",
                                                 "ll",
                                                 FD,
                                                 Path);
    revng_assert(!EC and FD != 0);
    llvm::raw_fd_ostream Stream(FD, true);
    F->print(Stream, nullptr);
    dbg << "Function printed to " << Path.str().str() << "\n";
    revng_abort();
  }
}

void collectTypes(Type *Root, std::set<Type *> &Set) {
  std::queue<Type *> ToVisit;
  ToVisit.push(Root);

  while (not ToVisit.empty()) {
    Type *T = ToVisit.front();
    ToVisit.pop();

    auto [_, IsNew] = Set.insert(T);
    if (not IsNew)
      continue;

    if (auto *Array = dyn_cast<ArrayType>(T)) {
      ToVisit.push(Array->getElementType());
    } else if (auto *FT = dyn_cast<FunctionType>(T)) {
      ToVisit.push(FT->getReturnType());
      for (Type *ParameterType : FT->params())
        ToVisit.push(ParameterType);
    } else if (isa<IntegerType>(T)) {
      // Nothing to do
    } else if (isa<PointerType>(T)) {
      // Nothing to do
    } else if (auto *Struct = dyn_cast<StructType>(T)) {
      for (Type *ElementType : Struct->elements())
        ToVisit.push(ElementType);
    } else if (auto *TET = dyn_cast<TargetExtType>(T)) {
      for (Type *TypeParam : TET->type_params())
        ToVisit.push(TypeParam);
    } else if (auto *TPT = dyn_cast<TypedPointerType>(T)) {
      ToVisit.push(TPT->getElementType());
    } else if (auto *Vector = dyn_cast<VectorType>(T)) {
      ToVisit.push(Vector->getElementType());
    } else {
      revng_abort();
    }
  }
}
