/// \file IRHelpers.cpp
/// Implementation of IR helper functions.

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <fstream>

#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/TypedPointerType.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Linker/Linker.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_os_ostream.h"

#include "revng/ADT/Queue.h"
#include "revng/ADT/RecursiveCoroutine.h"
#include "revng/Support/BlockType.h"
#include "revng/Support/IRHelpers.h"
#include "revng/Support/ProgramCounterHandler.h"
#include "revng/Support/StringOperations.h"
#include "revng/Support/Tag.h"

using namespace llvm;

void dumpModule(const Module *M, const char *Path) {
  std::ofstream FileStream(Path);
  raw_os_ostream Stream(FileStream);
  M->print(Stream, nullptr, false, true);
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

Constant *getUniqueString(Module *M, StringRef String, StringRef Namespace) {
  revng_assert(not Namespace.empty());

  LLVMContext &Context = M->getContext();
  std::string GlobalName = (Twine(Namespace) + revng::mangleName(String)).str();
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

Function &recreateWithoutBody(Function &OldFunction, FunctionType &NewType) {
  // Recreate the function as similar as possible
  auto *NewFunction = Function::Create(&NewType,
                                       GlobalValue::ExternalLinkage,
                                       "",
                                       OldFunction.getParent());
  NewFunction->takeName(&OldFunction);
  NewFunction->copyAttributesFrom(&OldFunction);
  NewFunction->copyMetadata(&OldFunction, 0);

  return *NewFunction;
}

Function &moveToNewFunctionType(Function &OldFunction, FunctionType &NewType) {
  Function &NewFunction = recreateWithoutBody(OldFunction, NewType);

  // Steal body
  if (not OldFunction.isDeclaration())
    moveBlocksInto(OldFunction, NewFunction);

  return NewFunction;
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
    Function *F = nullptr;
    BasicBlock *BB = nullptr;
    Instruction *I = nullptr;
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

    auto &&[_, IsNew] = Set.insert(T);
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

void pushInstructionALAP(llvm::DominatorTree &DT, llvm::Instruction *ToMove) {
  using namespace llvm;

  llvm::DenseSet<Instruction *> Users;
  BasicBlock *CommonDominator = nullptr;
  for (User *U : ToMove->users()) {
    if (auto *I = dyn_cast<Instruction>(U)) {
      Users.insert(I);
      auto *BB = I->getParent();
      if (CommonDominator == nullptr) {
        CommonDominator = BB;
      } else {
        CommonDominator = DT.findNearestCommonDominator(CommonDominator, BB);
      }
    }
  }

  revng_assert(CommonDominator != nullptr);

  for (Instruction &I : *CommonDominator) {
    if (I.isTerminator() or Users.contains(&I)) {
      ToMove->moveBefore(&I);
      return;
    }
  }

  revng_abort("Block has no terminator");
}

unsigned getMemoryAccessSize(llvm::Instruction *I) {
  using namespace llvm;
  Type *T = nullptr;

  if (auto *Load = dyn_cast<LoadInst>(I))
    T = Load->getType();
  else if (auto *Store = dyn_cast<StoreInst>(I))
    T = Store->getValueOperand()->getType();
  else
    revng_abort();

  return llvm::cast<llvm::IntegerType>(T)->getBitWidth() / 8;
}

bool deleteOnlyBody(llvm::Function &F) {
  bool Result = false;
  if (not F.empty()) {
    // deleteBody() also kills all attributes and tags. Since we still
    // want them, we have to save them and re-add them after deleting the
    // body of the function.
    auto Attributes = F.getAttributes();
    auto FTags = FunctionTags::TagsSet::from(&F);

    MetadataBackup SavedMetadata(&F);

    // Kill the body.
    F.deleteBody();

    // Restore tags and attributes
    FTags.set(&F);
    F.setAttributes(Attributes);

    F.clearMetadata();

    SavedMetadata.restoreIn(&F);

    Result = true;
  }
  return Result;
}

void sortModule(llvm::Module &M) {
  auto CompareByName = [](const auto *LHS, const auto *RHS) {
    return LHS->getName() < RHS->getName();
  };

  //
  // Reorder global variables
  //
  std::vector<llvm::GlobalVariable *> Globals;
  for (auto &Global : M.globals())
    Globals.push_back(&Global);

  for (auto *Global : Globals)
    Global->removeFromParent();

  llvm::sort(Globals, CompareByName);

  for (auto *Global : Globals)
    M.getGlobalList().push_back(Global);

  //
  // Reorder functions
  //
  std::vector<llvm::Function *> Functions;
  for (llvm::Function &F : M.functions())
    Functions.push_back(&F);

  for (llvm::Function *F : Functions)
    F->removeFromParent();

  std::sort(Functions.begin(), Functions.end(), CompareByName);

  for (llvm::Function *F : Functions)
    M.getFunctionList().push_back(F);

  //
  // Reorder basic blocks
  //
  for (llvm::Function *F : Functions) {
    if (F->isDeclaration() || F->empty())
      continue;

    llvm::BasicBlock *EntryBlock = &F->getEntryBlock();
    auto RPOT = llvm::ReversePostOrderTraversal(EntryBlock);

    llvm::SetVector<llvm::BasicBlock *> SortedBlocks;

    // Collect blocks in reverse port-order
    for (auto *BB : RPOT)
      SortedBlocks.insert(BB);

    // Collect blocks left out
    for (auto &BB : *F)
      if (not SortedBlocks.contains(&BB))
        SortedBlocks.insert(&BB);

    // Purge all the blocks from the function
    while (!F->empty())
      F->begin()->removeFromParent();

    // Re-add blocks in the correct order
    for (auto *BB : SortedBlocks)
      BB->insertInto(F);
  }
}

void linkModules(std::unique_ptr<llvm::Module> &&Source,
                 llvm::Module &Destination,
                 std::optional<llvm::GlobalValue::LinkageTypes> FinalLinkage) {
  std::map<std::string, llvm::GlobalValue::LinkageTypes> HelperGlobals;

  auto HandleGlobals = [&HelperGlobals, &Destination](auto &&GlobalsRange) {
    using T = std::decay_t<decltype(*GlobalsRange.begin())>;
    for (T &HelperGlobal : GlobalsRange) {
      auto GlobalName = HelperGlobal.getName();

      if (GlobalName.empty() or GlobalName.startswith("llvm."))
        continue;

      // Register so we can change its linkage later
      HelperGlobals[GlobalName.str()] = HelperGlobal.getLinkage();

      llvm::GlobalObject *LocalGlobal = nullptr;
      if constexpr (std::is_same_v<T, llvm::GlobalVariable>) {
        LocalGlobal = Destination.getGlobalVariable(GlobalName, true);
      } else {
        static_assert(std::is_same_v<T, llvm::Function>);
        LocalGlobal = Destination.getFunction(GlobalName);
      }

      if (LocalGlobal != nullptr) {
        // We have a global with the same name
        HelperGlobal.setLinkage(llvm::GlobalValue::ExternalLinkage);
        LocalGlobal->setLinkage(llvm::GlobalValue::ExternalLinkage);

        bool AlreadyAvailable = not LocalGlobal->isDeclaration();
        if (AlreadyAvailable) {
          // Turn helper global into declaration
          if constexpr (std::is_same_v<T, llvm::GlobalVariable>) {
            HelperGlobal.setInitializer(nullptr);
          } else {
            static_assert(std::is_same_v<T, llvm::Function>);
            HelperGlobal.deleteBody();
          }
        }
      }
    }
  };

  HandleGlobals(Source->globals());
  HandleGlobals(Source->functions());

  llvm::Linker TheLinker(Destination);
  bool Failed = TheLinker.linkInModule(std::move(Source),
                                       llvm::Linker::LinkOnlyNeeded);
  revng_assert(not Failed, "Linking failed");

  for (auto [GlobalName, Linkage] : HelperGlobals) {
    if (auto *GV = Destination.getGlobalVariable(GlobalName))
      if (not GV->isDeclaration())
        GV->setLinkage(FinalLinkage.value_or(Linkage));

    if (auto *F = Destination.getFunction(GlobalName))
      if (not F->isDeclaration())
        F->setLinkage(FinalLinkage.value_or(Linkage));
  }
}
