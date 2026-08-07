/// Implementation of IR helper functions.

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <fstream>
#include <optional>

#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/TypedPointerType.h"
#include "llvm/IR/Verifier.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Linker/Linker.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_os_ostream.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/ValueMapper.h"

#include "revng/ADT/Queue.h"
#include "revng/ADT/RecursiveCoroutine.h"
#include "revng/Model/FunctionTags.h"
#include "revng/Model/ProgramCounterHandler.h"
#include "revng/Support/BlockType.h"
#include "revng/Support/IRHelpers.h"
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

Constant *getUniqueString(Module *M,
                          StringRef String,
                          bool AddNull,
                          StringRef Namespace) {
  using revng::mangleName;

  revng_assert(not Namespace.empty());
  StringRef StringWithoutNUL = String.drop_back(AddNull ? 0 : 1);

  LLVMContext &Context = M->getContext();
  auto GlobalName = (Twine(Namespace) + mangleName(StringWithoutNUL)).str();

  // This may return a ConstantAggregateZero in case of empty String.
  Constant *Initializer = ConstantDataArray::getString(Context,
                                                       String,
                                                       AddNull);
  revng_assert(isa<ConstantDataArray>(Initializer)
               or isa<ConstantAggregateZero>(Initializer));
  if (String.empty()) {
    revng_assert(isa<ConstantAggregateZero>(Initializer));
  } else {
    auto CDAInitializer = cast<ConstantDataArray>(Initializer);
    auto Data = CDAInitializer->getRawDataValues().drop_back();
    revng_assert(Data == StringWithoutNUL);
  }

  auto *Global = &getOrCreateGlobal(*M,
                                    GlobalName,
                                    Initializer->getType(),
                                    true,
                                    GlobalValue::LinkOnceODRLinkage,
                                    Initializer);

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

void setNewPCOwner(Function *F, const MetaAddress &Owner) {
  Constant *Value = Owner.toValue(F->getParent());
  for (Instruction &I : instructions(F))
    if (CallInst *NewPCCall = getCallTo(&I, "newpc"))
      NewPCCall->setArgOperand(NewPCArguments::OwnerFunction, Value);
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

    MetadataBackup SavedMetadata(&F);

    // Kill the body.
    F.deleteBody();

    // Restore tags and attributes
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

std::unique_ptr<Module> parseIR(LLVMContext &Context, StringRef Path) {
  std::unique_ptr<Module> Result;
  SMDiagnostic Errors;
  Result = parseIRFile(Path, Errors, Context);

  if (Result.get() == nullptr) {
    Errors.print("revng", dbgs());
    revng_abort();
  }

  return Result;
}

void linkModules(std::unique_ptr<Module> &&Source,
                 Module &Destination,
                 std::optional<GlobalValue::LinkageTypes> FinalLinkage) {
  std::map<std::string, GlobalValue::LinkageTypes> HelperGlobals;

  auto HandleGlobals = [&HelperGlobals, &Destination](auto &&GlobalsRange) {
    using T = std::decay_t<decltype(*GlobalsRange.begin())>;
    for (T &HelperGlobal : GlobalsRange) {
      auto GlobalName = HelperGlobal.getName();

      if (not GlobalName.startswith("llvm.")) {

        // Register so we can change its linkage later
        HelperGlobals[GlobalName.str()] = HelperGlobal.getLinkage();

        GlobalObject *LocalGlobal = nullptr;
        if constexpr (std::is_same_v<T, GlobalVariable>) {
          LocalGlobal = Destination.getGlobalVariable(GlobalName);
        } else {
          static_assert(std::is_same_v<T, Function>);
          LocalGlobal = Destination.getFunction(GlobalName);
        }

        if (LocalGlobal != nullptr) {
          // We have a global with the same name
          HelperGlobal.setLinkage(GlobalValue::ExternalLinkage);

          bool AlreadyAvailable = not LocalGlobal->isDeclaration();
          if (AlreadyAvailable) {
            // Turn helper global into declaration
            if constexpr (std::is_same_v<T, GlobalVariable>) {
              HelperGlobal.setInitializer(nullptr);
            } else {
              static_assert(std::is_same_v<T, Function>);
              HelperGlobal.deleteBody();
            }
          } else {
            // Ensure it will be linked
            LocalGlobal->setLinkage(GlobalValue::ExternalLinkage);
          }
        }
      }
    }
  };

  HandleGlobals(Source->globals());
  HandleGlobals(Source->functions());

  Linker TheLinker(Destination);
  bool Failed = TheLinker.linkInModule(std::move(Source),
                                       Linker::LinkOnlyNeeded);
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

struct FunctionsMetadata {
  static inline const char *MetadataName = "revng.functions-metatadata-backup";

  static void dropBackup(llvm::Module &M) {
    M.eraseNamedMetadata(M.getNamedMetadata(MetadataName));
  }

  static void backup(llvm::Module &M) {
    using namespace llvm;

    LLVMContext &Context = M.getContext();

    // Create the named metadata
    NamedMDNode *BackupNMD = M.getOrInsertNamedMetadata(MetadataName);
    BackupNMD->clearOperands();

    // Backup saving names, the metadata kind IDs might change
    SmallVector<StringRef> MDKindNames;
    M.getMDKindNames(MDKindNames);

    for (Function &F : M) {
      // Ignore unnamed functions
      if (F.getName().empty())
        continue;

      // Collect metadata for the function
      SmallVector<std::pair<unsigned, MDNode *>, 8> MDs;
      F.getAllMetadata(MDs);

      // Skip if no metadata
      if (MDs.empty())
        continue;

      // Create an MDNode for the function's metadata
      // Format: [FunctionName, MDKind1, MDNode1, MDKind2, MDNode2, ...]
      SmallVector<Metadata *, 8> BackupEntry;
      BackupEntry.push_back(MDString::get(Context, F.getName()));

      for (const auto &MD : MDs) {
        BackupEntry.push_back(MDString::get(Context, MDKindNames[MD.first]));
        BackupEntry.push_back(MD.second);
      }

      // Append entry
      BackupNMD->addOperand(MDNode::get(Context, BackupEntry));
    }
  }

  static void restore(llvm::Module &M) {
    using namespace llvm;

    NamedMDNode *BackupNMD = M.getNamedMetadata(MetadataName);
    if (!BackupNMD)
      return;

    // Iterate over all operands in the backup NamedMDNode
    for (MDNode *N : BackupNMD->operands()) {
      if (N->getNumOperands() == 0)
        continue;

      // Get the function name
      StringRef FuncName = cast<MDString>(N->getOperand(0))->getString();

      // Find the function by name
      Function *F = M.getFunction(FuncName);
      if (F == nullptr)
        continue;

      // Clear existing metadata
      F->clearMetadata();

      auto OperandCount = N->getNumOperands();
      revng_assert(OperandCount >= 3 and OperandCount % 2 == 1);

      // Restore metadata (operands come in pairs: kind name, MDNode)
      for (unsigned I = 1; I < OperandCount; I += 2) {
        auto *KindMDString = cast<MDString>(N->getOperand(I).get());
        auto KindMD = M.getContext().getMDKindID(KindMDString->getString());
        MDNode *MD = cast<MDNode>(N->getOperand(I + 1));

        // llvm::verifyModule will fail if it encounters a function declaration
        // with a `!dbg` with a `distinct` MD. Since after filtering a function
        // might have become a declaration we skip setting `!dbg` it if that's
        // the case.
        if (F->isDeclaration() and KindMD == llvm::LLVMContext::MD_dbg)
          continue;

        F->setMetadata(KindMD, MD);
      }
    }
  }
};

std::unique_ptr<llvm::Module>
cloneFiltered(llvm::Module &Module, std::set<const llvm::Function *> &ToClone) {
  const auto Filter = [&ToClone](const auto &GlobalSym) {
    if (not llvm::isa<llvm::Function>(GlobalSym))
      return CloneAction::Clone;

    const auto &F = llvm::cast<llvm::Function>(GlobalSym);
    return ToClone.contains(F) ? CloneAction::Clone :
                                 CloneAction::MakeDeclaration;
  };

  return cloneFiltered(Module, Filter);
}

std::unique_ptr<llvm::Module>
cloneFiltered(llvm::Module &Module,
              llvm::function_ref<llvm::CloneAction(const llvm::GlobalValue *)>
                Action) {
  llvm::ValueToValueMapTy Map;

  FunctionsMetadata::backup(Module);

  revng::verify(&Module);
  auto Cloned = llvm::CloneModule(Module, Map, Action);

  FunctionsMetadata::restore(*Cloned.get());
  FunctionsMetadata::dropBackup(Module);
  FunctionsMetadata::dropBackup(*Cloned.get());

  return Cloned;
}

void writeBitcode(const llvm::Module &Module,
                  llvm::SmallVectorImpl<char> &Output) {
  llvm::BitcodeWriter Writer(Output);
  Writer.writeModule(Module);
  Writer.writeSymtab();
  Writer.writeStrtab();
}

std::unique_ptr<llvm::Module> readBitcode(llvm::ArrayRef<char> Input,
                                          llvm::LLVMContext &Context) {
  llvm::MemoryBufferRef BufferRef{ { Input.data(), Input.size() }, "input" };
  return llvm::cantFail(llvm::parseBitcodeFile(BufferRef, Context));
}

std::unique_ptr<llvm::Module> cloneIntoContext(const llvm::Module &Module,
                                               llvm::LLVMContext &NewContext) {
  revng_assert(&Module.getContext() != &NewContext);

  llvm::SmallVector<char, 0> Buffer;
  writeBitcode(Module, Buffer);
  return readBitcode(Buffer, NewContext);
}

/// Creates a global variable in the provided module that holds a pointer to
/// each other global object so that they can't be removed by the linker
static void makeGlobalObjectsArray(llvm::Module &Module,
                                   llvm::StringRef GlobalArrayName) {
  auto *PointerTy = llvm::PointerType::getUnqual(Module.getContext());

  llvm::SmallVector<llvm::Constant *, 10> Globals;

  for (auto &Global : Module.globals())
    Globals.push_back(llvm::ConstantExpr::getPointerCast(&Global, PointerTy));

  for (auto &Global : Module.functions())
    if (not Global.isIntrinsic())
      Globals.push_back(llvm::ConstantExpr::getPointerCast(&Global, PointerTy));

  auto *GlobalArrayType = llvm::ArrayType::get(PointerTy, Globals.size());

  auto *Initializer = llvm::ConstantArray::get(GlobalArrayType, Globals);

  new llvm::GlobalVariable(Module,
                           GlobalArrayType,
                           false,
                           llvm::GlobalValue::LinkageTypes::ExternalLinkage,
                           Initializer,
                           GlobalArrayName);
}

class GlobalsFixer {
private:
  static constexpr llvm::StringRef Prefix = "globals_fixer_variable_";

private:
  size_t Counter = 0;
  llvm::StringMap<llvm::GlobalValue::LinkageTypes> LinkageMap;
  llvm::StringMap<std::string> GlobalsRenameMap;

public:
  void recordGlobals(llvm::Module &Module) {
    using namespace llvm;

    for (auto &Global : Module.global_objects()) {
      bool IsUniqued = isUniquedGlobal(Global);

      // Rename declarations so they are not merged by the linker
      if (Global.isDeclaration() and IsUniqued) {
        llvm::StringRef OldName = Global.getName();
        std::string NewName = Prefix.str() + std::to_string(Counter);
        GlobalsRenameMap[NewName] = OldName.str();
        Global.setName(NewName);
        Counter++;
      }

      // Turn globals with local linkage and external declarations into the
      // equivalent of inline and record their original linking for it be
      // restored later
      if (Global.getLinkage() == GlobalValue::InternalLinkage
          or Global.getLinkage() == GlobalValue::PrivateLinkage
          or Global.getLinkage() == GlobalValue::AppendingLinkage
          or (Global.getLinkage() == GlobalValue::ExternalLinkage
              and not Global.isDeclaration())) {
        LinkageMap[Global.getName()] = Global.getLinkage();

        // Globals tagged with UniquedByPrototype and UniquedByMetadata are
        // deduplicated manually post-link, mark them with Internal linkage so
        // they are not collapsed by the linker
        if (IsUniqued)
          Global.setLinkage(GlobalValue::InternalLinkage);
        else
          Global.setLinkage(GlobalValue::LinkOnceODRLinkage);
      }
    }
  }

  void restoreGlobals(llvm::Module &Module) {
    // Restores the initial linkage for local functions
    for (auto &Global : Module.global_objects()) {
      auto It = LinkageMap.find(Global.getName().str());
      if (It != LinkageMap.end())
        Global.setLinkage(It->second);
    }

    // Restore the original name of the declaration
    for (auto &Global : Module.global_objects()) {
      if (not(Global.isDeclaration() and isUniquedGlobal(Global)))
        continue;

      auto OldNameEntry = GlobalsRenameMap.find(Global.getName());
      if (OldNameEntry == GlobalsRenameMap.end())
        continue;

      std::string OldName = OldNameEntry->second;
      Global.setName(OldNameEntry->second);
    }
  }

private:
  static bool isUniquedGlobal(llvm::GlobalObject &Object) {
    if (auto *Function = dyn_cast<llvm::Function>(&Object)) {
      return FunctionTags::UniquedByPrototype.isTagOf(Function)
             or FunctionTags::UniquedByMetadata.isTagOf(Function);
    } else if (auto *GlobalVariable = dyn_cast<llvm::GlobalVariable>(&Object)) {
      return FunctionTags::UniquedByPrototype.isTagOf(GlobalVariable)
             or FunctionTags::UniquedByMetadata.isTagOf(GlobalVariable);
    } else {
      return false;
    }
  }
};

void linkFunctionModules(std::unique_ptr<llvm::Module> &&Source,
                         std::unique_ptr<llvm::Module> &Destination) {
  GlobalsFixer Fixer;
  Fixer.recordGlobals(*Source);
  Fixer.recordGlobals(*Destination);

  // Make a global array of all global objects so that they don't get dropped
  std::string GlobalArray1 = "revng.AllSymbolsArrayLeft";
  makeGlobalObjectsArray(*Destination, GlobalArray1);

  std::string GlobalArray2 = "revng.AllSymbolsArrayRight";
  makeGlobalObjectsArray(*Source, GlobalArray2);

  // Drop certain LLVM named metadata
  auto DropNamedMetadata = [](llvm::Module *M, llvm::StringRef Name) {
    if (auto *MD = M->getNamedMetadata(Name))
      MD->eraseFromParent();
  };

  // TODO: check it's identical to the existing one, if present in both
  DropNamedMetadata(&*Destination, "llvm.ident");
  DropNamedMetadata(&*Destination, "llvm.module.flags");

  if (Source->getDataLayout().isDefault())
    Source->setDataLayout(Destination->getDataLayout());

  if (Destination->getDataLayout().isDefault())
    Destination->setDataLayout(Source->getDataLayout());

  llvm::Linker TheLinker(*Source);

  // Actually link
  bool Failure = TheLinker.linkInModule(std::move(Destination));

  revng_assert(not Failure, "Linker failed");

  Fixer.restoreGlobals(*Source);
  Destination = std::move(Source);

  // Remove the global arrays since they are no longer needed.
  if (auto *Global = Destination->getGlobalVariable(GlobalArray1))
    Global->eraseFromParent();

  if (auto *Global = Destination->getGlobalVariable(GlobalArray2))
    Global->eraseFromParent();

  llvm::DenseSet<llvm::Function *> ToErase;

  auto MarkDuplicates = [&ToErase](const auto &Map) {
    using Key = typename std::decay_t<decltype(Map)>::key_type;
    Key LastKey;
    llvm::Function *Leader = nullptr;
    for (auto &[Key, F] : Map) {

      if (Key != LastKey) {
        Leader = F;
        LastKey = Key;
      } else {
        revng_assert(F->getFunctionType() == Leader->getFunctionType());
        F->replaceAllUsesWith(Leader);
        ToErase.insert(F);
      }
    }
  };

  // Dedup based on UniquedByPrototype
  {
    using namespace llvm;
    using Key = std::pair<FunctionTags::TagsSet, FunctionType *>;

    std::multimap<Key, Function *> Map;
    for (Function &F :
         FunctionTags::UniquedByPrototype.functions(&*Destination)) {
      Key TheKey = { FunctionTags::TagsSet::from(&F), F.getFunctionType() };
      Map.emplace(TheKey, &F);
    }

    MarkDuplicates(Map);
  }

  // Dedup based on UniquedByMetadata
  {
    using namespace llvm;
    using Key = std::pair<FunctionTags::TagsSet, MDNode *>;

    std::multimap<Key, Function *> Map;
    for (Function &F :
         FunctionTags::UniquedByMetadata.functions(&*Destination)) {
      MDNode *MD = F.getMetadata(FunctionTags::UniqueIDMDName);
      revng_assert(MD->isUniqued());
      Key TheKey = { FunctionTags::TagsSet::from(&F), MD };
      Map.emplace(TheKey, &F);
    }

    MarkDuplicates(Map);
  }

  // Purge all unused non-target functions
  // TODO: this should be transitive
  for (llvm::Function &F : Destination->functions())
    if (not FunctionTags::Isolated.isTagOf(&F) and F.use_empty())
      ToErase.insert(&F);

  for (llvm::Function *F : ToErase)
    F->eraseFromParent();

  // Prune llvm.dbg.cu so that they grow exponentially due to multiple cloning
  // + linking.
  // Note: an alternative approach would be to pre-populate the
  //       ValueToValueMap used when we clone in a way that avoids cloning the
  //       metadata altogether. However, this would lead two distinct modules
  //       to share debug metadata, which are not always immutable.
  auto *NamedMDNode = Destination->getOrInsertNamedMetadata("llvm.dbg.cu");
  pruneDICompileUnits(*Destination);

  revng::verify(&*Destination);
}

template<bool IgnoreInitializrMismatch>
llvm::GlobalVariable &
getOrCreateGlobalInternal(llvm::Module &M,
                          llvm::StringRef Name,
                          llvm::Type *Type,
                          bool IsConstant,
                          llvm::GlobalValue::LinkageTypes Linkage,
                          llvm::Constant *Initializer) {
  revng_assert(not Name.empty());
  revng_assert(Name.data() != nullptr);
  bool IsInternal = Linkage == llvm::GlobalValue::InternalLinkage;
  llvm::GlobalVariable *Result = M.getGlobalVariable(Name, IsInternal);

  // Check if we already have this
  if (Result != nullptr) {
    // Compare features
    revng_assert(Result->getLinkage() == Linkage);
    revng_assert(Result->getValueType() == Type);
    revng_assert(Result->isConstant() == IsConstant);
    revng_assert(Result->isDeclaration() == (Initializer == nullptr));
    if (not Result->isDeclaration() and not IgnoreInitializrMismatch)
      revng_assert(Result->getInitializer() == Initializer);
    return *Result;
  }

  Result = new llvm::GlobalVariable(M,
                                    Type,
                                    IsConstant,
                                    Linkage,
                                    Initializer,
                                    Name);

  if (not IsInternal)
    revng_assert(Result->getName() == Name);

  return *Result;
}

llvm::GlobalVariable &getOrCreateGlobal(llvm::Module &M,
                                        llvm::StringRef Name,
                                        llvm::Type *Type,
                                        bool IsConstant,
                                        llvm::GlobalValue::LinkageTypes Linkage,
                                        llvm::Constant *Initializer) {
  return getOrCreateGlobalInternal<false>(M,
                                          Name,
                                          Type,
                                          IsConstant,
                                          Linkage,
                                          Initializer);
}

llvm::GlobalVariable &
getOrCreateUnstableGlobal(llvm::Module &M,
                          llvm::StringRef Name,
                          llvm::Type *Type,
                          bool IsConstant,
                          llvm::GlobalValue::LinkageTypes Linkage,
                          llvm::Constant *Initializer) {
  return getOrCreateGlobalInternal<true>(M,
                                         Name,
                                         Type,
                                         IsConstant,
                                         Linkage,
                                         Initializer);
}

void serializeInliningPolicy(llvm::Function &Helper,
                             const llvm::BitVector &CriticalArguments) {
  QuickMetadata QMD(Helper.getContext());

  // Use `arg_size() + 1` bits so the MSB is always zero. LLVM's IR printer
  // formats `iN` (N >= 2) constants as signed decimal; the extra bit keeps
  // the rendered value positive regardless of which arguments are critical. (No
  // `-2` value appears for a function whose only first argument is critical).
  unsigned Width = static_cast<unsigned>(Helper.arg_size()) + 1;
  llvm::APInt Bits(Width, 0);
  for (int Idx = CriticalArguments.find_first(); Idx >= 0;
       Idx = CriticalArguments.find_next(Idx))
    Bits.setBit(Idx);
  Helper.setMetadata(InliningPolicyMetadataKey, QMD.tuple({ QMD.get(Bits) }));
}

InliningPolicy deserializeInliningPolicy(const llvm::Function &Helper) {
  llvm::MDNode *MD = Helper.getMetadata(InliningPolicyMetadataKey);
  revng_assert(MD != nullptr,
               "revng_inline helper missing `revng.inline.policy` metadata");

  QuickMetadata QMD(Helper.getContext());
  auto *Bits = QMD.extract<llvm::ConstantInt *>(llvm::cast<llvm::MDTuple>(MD),
                                                0);
  const llvm::APInt &Value = Bits->getValue();

  InliningPolicy Result;
  Result.CriticalArguments = llvm::BitVector(Helper.arg_size(), false);
  for (unsigned I = 0; I < Helper.arg_size(); ++I)
    if (Value[I])
      Result.CriticalArguments.set(I);
  return Result;
}
