/// \file IRHelpers.cpp
/// Implementation of IR helper functions.

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <fstream>

#include "llvm/ADT/SmallSet.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/TypedPointerType.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/SHA1.h"
#include "llvm/Support/raw_os_ostream.h"

#include "revng/ADT/Queue.h"
#include "revng/ADT/RecursiveCoroutine.h"
#include "revng/Support/BlockType.h"
#include "revng/Support/IRHelpers.h"
#include "revng/Support/ProgramCounterHandler.h"

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

void emitCall(llvm::IRBuilderBase &Builder,
              Function *Callee,
              const Twine &Reason,
              const DebugLoc &DbgLocation,
              const ProgramCounterHandler *PCH) {
  revng_assert(Callee != nullptr);
  llvm::Module *M = Callee->getParent();

  SmallVector<llvm::Value *, 4> Arguments;

  // Create the message string
  Arguments.push_back(getUniqueString(M, Reason.str()));

  // Populate the source PC
  MetaAddress SourcePC = MetaAddress::invalid();

  if (Instruction *T = Builder.GetInsertBlock()->getTerminator())
    SourcePC = getPC(T).first;

  if (PCH != nullptr) {
    PCH->setLastPCPlainMetaAddress(Builder, SourcePC);
    PCH->setCurrentPCPlainMetaAddress(Builder);
  }

  auto *NewCall = Builder.CreateCall(Callee, Arguments);
  NewCall->setDebugLoc(DbgLocation);
  Builder.CreateUnreachable();

  // Assert there's one and only one terminator
  auto *BB = Builder.GetInsertBlock();
  unsigned Terminators = 0;
  for (Instruction &I : *BB)
    if (I.isTerminator())
      ++Terminators;
  revng_assert(Terminators == 1);
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

template<typename T>
concept DerivedValue = std::is_base_of_v<llvm::Value, T>;

using std::conditional_t;

template<DerivedValue ConstnessT, DerivedValue ResultT>
using PossiblyConstValueT = conditional_t<std::is_const_v<ConstnessT>,
                                          std::add_const_t<ResultT>,
                                          std::remove_const_t<ResultT>>;

template<DerivedValue T>
using ValueT = PossiblyConstValueT<T, llvm::Value>;

template<DerivedValue T>
using CallT = PossiblyConstValueT<T, llvm::CallInst>;

template<DerivedValue T>
using CallPtrSet = llvm::SmallPtrSet<CallT<T> *, 2>;

template<DerivedValue T>
llvm::SmallVector<CallPtrSet<T>, 2>
getConstQualifiedExtractedValuesFromInstruction(T *I) {

  llvm::SmallVector<CallPtrSet<T>, 2> Results;

  auto *StructTy = llvm::cast<llvm::StructType>(I->getType());
  unsigned NumFields = StructTy->getNumElements();
  Results.resize(NumFields, {});

  // Find extract value uses transitively, traversing PHIs and markers
  CallPtrSet<T> Calls;
  for (auto *TheUser : I->users()) {
    if (auto *ExtractV = getCallToTagged(TheUser,
                                         FunctionTags::OpaqueExtractValue)) {
      Calls.insert(ExtractV);
    } else {
      if (auto *Call = dyn_cast<llvm::CallInst>(TheUser)) {
        if (not isCallToTagged(Call, FunctionTags::Parentheses))
          continue;
      }

      // traverse PHIS and markers until we find extractvalues
      llvm::SmallPtrSet<ValueT<T> *, 8> Visited = {};
      llvm::SmallPtrSet<ValueT<T> *, 8> ToVisit = { TheUser };
      while (not ToVisit.empty()) {

        llvm::SmallPtrSet<ValueT<T> *, 8> NextToVisit = {};

        for (ValueT<T> *Ident : ToVisit) {
          Visited.insert(Ident);
          NextToVisit.erase(Ident);

          for (auto *User : Ident->users()) {
            using FunctionTags::OpaqueExtractValue;
            if (auto *EV = getCallToTagged(User, OpaqueExtractValue)) {
              Calls.insert(EV);
            } else if (auto *IdentUser = llvm::dyn_cast<llvm::CallInst>(User)) {
              if (isCallToTagged(IdentUser, FunctionTags::Parentheses))
                NextToVisit.insert(IdentUser);
            } else if (auto *PHIUser = llvm::dyn_cast<llvm::PHINode>(User)) {
              if (not Visited.contains(PHIUser))
                NextToVisit.insert(PHIUser);
            }
          }
        }

        ToVisit = NextToVisit;
      }
    }
  }

  for (auto *E : Calls) {
    revng_assert(isa<llvm::IntegerType>(E->getType())
                 or isa<llvm::PointerType>(E->getType()));
    auto FieldId = cast<llvm::ConstantInt>(E->getArgOperand(1))->getZExtValue();
    Results[FieldId].insert(E);
  }

  return Results;
};

llvm::SmallVector<llvm::SmallPtrSet<llvm::CallInst *, 2>, 2>
getExtractedValuesFromInstruction(llvm::Instruction *I) {
  return getConstQualifiedExtractedValuesFromInstruction(I);
}

llvm::SmallVector<llvm::SmallPtrSet<const llvm::CallInst *, 2>, 2>
getExtractedValuesFromInstruction(const llvm::Instruction *I) {
  return getConstQualifiedExtractedValuesFromInstruction(I);
}

bool deleteOnlyBody(llvm::Function &F) {
  bool Result = false;
  if (not F.empty()) {
    // deleteBody() also kills all attributes and tags. Since we still
    // want them, we have to save them and re-add them after deleting the
    // body of the function.
    auto Attributes = F.getAttributes();
    auto FTags = FunctionTags::TagsSet::from(&F);

    llvm::SmallVector<std::pair<unsigned, llvm::MDNode *>> AllMetadata;
    if (F.hasMetadata())
      F.getAllMetadata(AllMetadata);

    // Kill the body.
    F.deleteBody();

    // Restore tags and attributes
    FTags.set(&F);
    F.setAttributes(Attributes);

    F.clearMetadata();
    for (const auto &[KindID, MetaData] : AllMetadata) {
      // Debug metadata is not stripped away by deleteBody() nor by
      // clearMetadata(), but it is wrong to set it twice (the Module would not
      // verify anymore). Hence set the metadata only if its not a debug
      // metadata.
      if (not F.hasMetadata(KindID) and KindID != llvm::LLVMContext::MD_dbg)
        F.setMetadata(KindID, MetaData);
    }

    Result = true;
  }
  return Result;
}

void setSegmentKeyMetadata(llvm::Function &SegmentRefFunction,
                           MetaAddress StartAddress,
                           uint64_t VirtualSize) {
  using namespace llvm;

  auto &Context = SegmentRefFunction.getContext();

  QuickMetadata QMD(Context);

  auto *SAMD = QMD.get(StartAddress.toString());
  revng_assert(SAMD != nullptr);

  auto *VSConstant = ConstantInt::get(Type::getInt64Ty(Context), VirtualSize);
  auto *VSMD = ConstantAsMetadata::get(VSConstant);

  SegmentRefFunction.setMetadata(FunctionTags::UniqueIDMDName,
                                 QMD.tuple({ SAMD, VSMD }));
}

bool hasSegmentKeyMetadata(const llvm::Function &F) {
  auto &Context = F.getContext();
  auto SegmentRefMDKind = Context.getMDKindID(FunctionTags::UniqueIDMDName);
  return nullptr != F.getMetadata(SegmentRefMDKind);
}

std::pair<MetaAddress, uint64_t>
extractSegmentKeyFromMetadata(const llvm::Function &F) {
  using namespace llvm;
  revng_assert(hasSegmentKeyMetadata(F));

  auto &Context = F.getContext();

  auto SegmentRefMDKind = Context.getMDKindID(FunctionTags::UniqueIDMDName);
  auto *Node = F.getMetadata(SegmentRefMDKind);

  auto *SAMD = cast<MDString>(Node->getOperand(0));
  MetaAddress StartAddress = MetaAddress::fromString(SAMD->getString());
  auto *VSMD = cast<ConstantAsMetadata>(Node->getOperand(1))->getValue();
  uint64_t VirtualSize = cast<ConstantInt>(VSMD)->getZExtValue();

  return { StartAddress, VirtualSize };
}

void setStringLiteralMetadata(llvm::Function &StringLiteralFunction,
                              MetaAddress StartAddress,
                              uint64_t VirtualSize,
                              uint64_t Offset,
                              uint64_t StringLength,
                              llvm::Type *ReturnType) {
  using namespace llvm;

  auto *M = StringLiteralFunction.getParent();
  auto &Context = StringLiteralFunction.getContext();

  QuickMetadata QMD(M->getContext());
  auto StringLiteralMDKind = Context.getMDKindID(FunctionTags::UniqueIDMDName);

  auto *SAMD = QMD.get(StartAddress.toString());

  auto *VSConstant = ConstantInt::get(Type::getInt64Ty(Context), VirtualSize);
  auto *VSMD = ConstantAsMetadata::get(VSConstant);

  auto *OffsetConstant = ConstantInt::get(Type::getInt64Ty(Context), Offset);
  auto *OffsetMD = ConstantAsMetadata::get(OffsetConstant);

  auto *StrLenConstant = ConstantInt::get(Type::getInt64Ty(Context),
                                          StringLength);
  auto *StrLenMD = ConstantAsMetadata::get(StrLenConstant);

  unsigned Value = ReturnType->isPointerTy() ? 0 :
                                               ReturnType->getIntegerBitWidth();
  auto *ReturnTypeConstant = ConstantInt::get(Type::getInt64Ty(Context), Value);
  auto *ReturnTypeMD = ConstantAsMetadata::get(ReturnTypeConstant);

  auto QMDTuple = QMD.tuple({ SAMD, VSMD, OffsetMD, StrLenMD, ReturnTypeMD });
  StringLiteralFunction.setMetadata(StringLiteralMDKind, QMDTuple);
}

bool hasStringLiteralMetadata(const llvm::Function &F) {
  auto &Context = F.getContext();
  auto StringLiteralMDKind = Context.getMDKindID(FunctionTags::UniqueIDMDName);
  return nullptr != F.getMetadata(StringLiteralMDKind);
}

std::tuple<MetaAddress, uint64_t, uint64_t, uint64_t, llvm::Type *>
extractStringLiteralFromMetadata(const llvm::Function &F) {
  using namespace llvm;
  revng_assert(hasStringLiteralMetadata(F));

  auto &Context = F.getContext();

  auto StringLiteralMDKind = Context.getMDKindID(FunctionTags::UniqueIDMDName);
  auto *Node = F.getMetadata(StringLiteralMDKind);

  StringRef SAMD = cast<MDString>(Node->getOperand(0))->getString();
  MetaAddress StartAddress = MetaAddress::fromString(SAMD);

  auto ExtractInteger = [](const MDOperand &Operand) {
    auto *MD = cast<ConstantAsMetadata>(Operand)->getValue();
    return cast<ConstantInt>(MD)->getZExtValue();
  };

  uint64_t VirtualSize = ExtractInteger(Node->getOperand(1));
  uint64_t Offset = ExtractInteger(Node->getOperand(2));
  uint64_t StrLen = ExtractInteger(Node->getOperand(3));
  uint64_t ReturnTypeLength = ExtractInteger(Node->getOperand(4));
  llvm::Type *PointerType = llvm::PointerType::get(Context, 0);
  llvm::Type *ReturnType = ReturnTypeLength == 0 ?
                             PointerType :
                             llvm::IntegerType::get(Context, ReturnTypeLength);

  return { StartAddress, VirtualSize, Offset, StrLen, ReturnType };
}

void emitMessage(llvm::Instruction *EmitBefore, const llvm::Twine &Message) {
  llvm::IRBuilder<> Builder(EmitBefore);
  emitMessage(Builder, Message);
}

void emitMessage(llvm::IRBuilder<> &Builder, const llvm::Twine &Message) {
  using namespace llvm;

  Module *M = getModule(Builder.GetInsertBlock());
  auto *FT = createFunctionType<void, const uint8_t *>(M->getContext());
  // TODO: use reserved prefix
  llvm::StringRef MessageFunctionName("revng_message");
  FunctionCallee Callee = M->getOrInsertFunction(MessageFunctionName, FT);

  Function *F = cast<Function>(Callee.getCallee());
  if (not FunctionTags::Helper.isTagOf(F))
    FunctionTags::Helper.addTo(F);

  Builder.CreateCall(Callee, getUniqueString(M, "emitMessage", Message.str()));
}
