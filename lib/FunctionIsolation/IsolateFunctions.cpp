/// \file IsolateFunctions.cpp
/// \brief Implements the IsolateFunctions pass which applies function isolation
///        using the informations provided by FunctionBoundariesDetectionPass.

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// LLVM includes
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/raw_os_ostream.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/Local.h"

// Local libraries includes
#include "revng/BasicAnalyses/GeneratedCodeBasicInfo.h"
#include "revng/FunctionIsolation/IsolateFunctions.h"
#include "revng/Runtime/commonconstants.h"
#include "revng/StackAnalysis/FunctionsSummary.h"
#include "revng/Support/Debug.h"
#include "revng/Support/IRHelpers.h"

using namespace llvm;

class IsolateFunctionsImpl;

// Define an alias for the data structure that will contain the LLVM functions
using FunctionsMap = std::map<MDString *, Function *>;

typedef DenseMap<const Value *, Value *> ValueToValueMap;

using IF = IsolateFunctions;
using IFI = IsolateFunctionsImpl;

char IF::ID = 0;
static RegisterPass<IF> X("isolate", "Isolate Functions Pass", true, true);

class IsolateFunctionsImpl {
private:
  struct IsolatedFunctionDescriptor {
    uint64_t PC;
    Function *IsolatedFunction;
    ValueToValueMap ValueMap;
    std::map<BasicBlock *, BasicBlock *> Trampolines;
    using BranchTypesMap = std::map<BasicBlock *,
                                    StackAnalysis::BranchType::Values>;
    BranchTypesMap Members;
    std::map<BasicBlock *, BasicBlock *> FakeReturnPaths;
  };

public:
  IsolateFunctionsImpl(Function *RootFunction, GeneratedCodeBasicInfo &GCBI) :
    RootFunction(RootFunction),
    TheModule(RootFunction->getParent()),
    GCBI(GCBI),
    Context(getContext(TheModule)),
    PCBitSize(8 * GCBI.pcRegSize()) {}

  void run();

private:
  /// \brief Creates the call that simulates the throw of an exception
  void throwException(Reason Code, BasicBlock *BB, uint64_t AdditionalPC);

  /// \brief Instantiate a basic block that consists only of an exception throw
  BasicBlock *createUnreachableBlock(StringRef Name, Function *CurrentFunction);

  /// \brief Populate the @function_dispatcher, needed to handle the indirect
  ///        function calls
  void populateFunctionDispatcher();

  /// \brief Create the basic blocks that are hit on exit after an invoke
  ///        instruction
  BasicBlock *createInvokeReturnBlock(Function *Root, BasicBlock *UnexpectedPC);

  /// \brief Create the basic blocks that represent the catch of the invoke
  ///        instruction
  BasicBlock *createCatchBlock(Function *Root, BasicBlock *UnexpectedPC);

  /// \brief Replace the call to the @function_call marker with the actual call
  ///
  /// \return true if the function call has been emitted
  bool replaceFunctionCall(BasicBlock *NewBB,
                           CallInst *Call,
                           const ValueToValueMap &RootToIsolated);

  /// \brief Checks if an instruction is a terminator with an invalid successor
  bool isTerminatorWithInvalidTarget(Instruction *I,
                                     const ValueToValueMap &RootToIsolated);

  /// \brief Handle the cloning of an instruction in the new basic block
  ///
  /// \return true if the function purged all the instructions after this one in
  ///         the current basic block
  bool cloneInstruction(BasicBlock *NewBB,
                        Instruction *OldInstruction,
                        IsolatedFunctionDescriptor &Descriptor);

  /// \brief Extract the string representing a function name starting from the
  ///        MDNode
  /// \return StringRef representing the function name
  StringRef getFunctionNameString(MDNode *Node);

private:
  Function *RootFunction;
  Module *TheModule;
  GeneratedCodeBasicInfo &GCBI;
  LLVMContext &Context;
  Function *RaiseException;
  Function *DebugException;
  Function *FunctionDispatcher;
  std::map<MDString *, IsolatedFunctionDescriptor> Functions;
  std::map<BasicBlock *, BasicBlock *> IsolatedToRootBB;
  GlobalVariable *ExceptionFlag;
  GlobalVariable *PC;
  const unsigned PCBitSize;
};

void IFI::throwException(Reason Code, BasicBlock *BB, uint64_t AdditionalPC) {
  revng_assert(PC != nullptr);
  revng_assert(RaiseException != nullptr);
  revng_assert(DebugException != nullptr);

  // Create a builder object
  IRBuilder<> Builder(Context);
  Builder.SetInsertPoint(BB);

  // Set the exception flag to value one
  ConstantInt *ConstantTrue = Builder.getTrue();
  Builder.CreateStore(ConstantTrue, ExceptionFlag);

  // Call the _debug_exception function to print usefull stuff
  LoadInst *ProgramCounter = Builder.CreateLoad(PC, "");

  uint64_t LastPC;

  if (Code == StandardTranslatedBlock) {
    // Retrieve the value of the PC in the basic block where the exception has
    // been raised, this is possible since BB should be a translated block
    LastPC = GCBI.getPC(&*BB->rbegin()).first;
    revng_assert(LastPC != 0);
  } else {

    // The current basic block has not been translated from the original binary
    // (e.g. unexpectedpc or anypc), therefore we can't retrieve the
    // corresponding PC.
    LastPC = 0;
  }

  // Get the PC register dimension and use it to instantiate the arguments of
  // the call to exception_warning
  ConstantInt *ReasonValue = Builder.getInt32(Code);
  ConstantInt *ConstantLastPC = Builder.getIntN(PCBitSize, LastPC);
  ConstantInt *ConstantAdditionalPC = Builder.getIntN(PCBitSize, AdditionalPC);

  // Emit the call to exception_warning
  Builder.CreateCall(DebugException,
                     { ReasonValue,
                       ConstantLastPC,
                       ProgramCounter,
                       ConstantAdditionalPC },
                     "");

  // Emit the call to _Unwind_RaiseException
  Builder.CreateCall(RaiseException);
  Builder.CreateUnreachable();
}

BasicBlock *
IFI::createUnreachableBlock(StringRef Name, Function *CurrentFunction) {

  // Create the basic block and add it in the function passed as parameter
  BasicBlock *NewBB = BasicBlock::Create(Context,
                                         Name,
                                         CurrentFunction,
                                         nullptr);

  throwException(StandardNonTranslatedBlock, NewBB, 0);
  return NewBB;
}

void IFI::populateFunctionDispatcher() {

  BasicBlock *DispatcherBB = BasicBlock::Create(Context,
                                                "function_dispatcher",
                                                FunctionDispatcher,
                                                nullptr);

  BasicBlock *UnexpectedPC = BasicBlock::Create(Context,
                                                "unexpectedpc",
                                                FunctionDispatcher,
                                                nullptr);
  throwException(FunctionDispatcherFallBack, UnexpectedPC, 0);

  // Create a builder object for the DispatcherBB basic block
  IRBuilder<> Builder(Context);
  Builder.SetInsertPoint(DispatcherBB);

  LoadInst *ProgramCounter = Builder.CreateLoad(PC, "");

  SwitchInst *Switch = Builder.CreateSwitch(ProgramCounter, UnexpectedPC);

  for (auto &Pair : Functions) {
    IsolatedFunctionDescriptor &Descriptor = Pair.second;
    Function *Function = Descriptor.IsolatedFunction;
    StringRef Name = Function->getName();

    // Creation of a basic block correspondent to the trampoline for each
    // function
    BasicBlock *TrampolineBB = BasicBlock::Create(Context,
                                                  Name + "_trampoline",
                                                  FunctionDispatcher,
                                                  nullptr);

    CallInst::Create(Function, "", TrampolineBB);
    ReturnInst::Create(Context, TrampolineBB);

    auto *Label = Builder.getIntN(PCBitSize, Descriptor.PC);
    Switch->addCase(Label, TrampolineBB);
  }
}

BasicBlock *
IFI::createInvokeReturnBlock(Function *Root, BasicBlock *UnexpectedPC) {

  // Create the first block
  BasicBlock *InvokeReturnBlock = BasicBlock::Create(Context,
                                                     "invoke_return",
                                                     Root,
                                                     nullptr);

  // Create two basic blocks, one that we will hit if we have a normal exit
  // from the invoke call and another for signaling the creation of an
  // exception, and connect both of them to the unexpectedpc block
  BasicBlock *NormalInvoke = BasicBlock::Create(Context,
                                                "normal_invoke",
                                                Root,
                                                nullptr);
  BranchInst::Create(UnexpectedPC, NormalInvoke);

  BasicBlock *AbnormalInvoke = BasicBlock::Create(Context,
                                                  "abnormal_invoke",
                                                  Root,
                                                  nullptr);

  // Create a builder object for the AbnormalInvokeReturn basic block
  IRBuilder<> BuilderAbnormalBB(Context);
  BuilderAbnormalBB.SetInsertPoint(AbnormalInvoke);

  ConstantInt *ConstantFalse = BuilderAbnormalBB.getFalse();
  BuilderAbnormalBB.CreateStore(ConstantFalse, ExceptionFlag);
  BuilderAbnormalBB.CreateBr(UnexpectedPC);

  // Create a builder object for the InvokeReturnBlock basic block
  IRBuilder<> BuilderReturnBB(Context);
  BuilderReturnBB.SetInsertPoint(InvokeReturnBlock);

  // Add a conditional branch at the end of the invoke exit block that jumps to
  // the right basic block on the basis of the flag.
  LoadInst *Flag = BuilderReturnBB.CreateLoad(ExceptionFlag, "");
  BuilderReturnBB.CreateCondBr(Flag, AbnormalInvoke, NormalInvoke);

  return InvokeReturnBlock;
}

BasicBlock *IFI::createCatchBlock(Function *Root, BasicBlock *UnexpectedPC) {

  // Create a basic block that represents the catch part of the exception
  BasicBlock *CatchBB = BasicBlock::Create(Context,
                                           "catchblock",
                                           Root,
                                           nullptr);

  // Create a builder object
  IRBuilder<> Builder(Context);
  Builder.SetInsertPoint(CatchBB);

  // Create the StructType necessary for the landingpad
  PointerType *RetTyPointerType = Type::getInt8PtrTy(Context);
  IntegerType *RetTyIntegerType = Type::getInt32Ty(Context);
  std::vector<Type *> InArgsType{ RetTyPointerType, RetTyIntegerType };
  StructType *RetTyStruct = StructType::create(Context,
                                               ArrayRef<Type *>(InArgsType),
                                               "",
                                               false);

  // Create the landingpad instruction
  LandingPadInst *LandingPad = Builder.CreateLandingPad(RetTyStruct, 0);

  // Add a catch all (constructed with the null value as clause)
  LandingPad->addClause(ConstantPointerNull::get(Type::getInt8PtrTy(Context)));

  // This should be an unreachable (we should never reach the catch block), but
  // to avoid optimizations that purge this basic block (and also the
  // correlated invoke instructions) we need a fake ret here
  Builder.CreateCall(TheModule->getFunction("abort"));
  Builder.CreateRetVoid();

  return CatchBB;
}

bool IFI::replaceFunctionCall(BasicBlock *NewBB,
                              CallInst *Call,
                              const ValueToValueMap &RootToIsolated) {

  // Retrieve the called function and emit the call
  BlockAddress *Callee = dyn_cast<BlockAddress>(Call->getOperand(0));
  bool IsIndirect = (Callee == nullptr);
  Function *TargetFunction = nullptr;

  if (IsIndirect) {
    TargetFunction = FunctionDispatcher;
  } else {
    BasicBlock *CalleeEntry = Callee->getBasicBlock();
    TerminatorInst *Terminator = CalleeEntry->getTerminator();
    auto *Node = cast<MDTuple>(Terminator->getMetadata("func.entry"));

    // The callee is not a real function, it must be part of us then.
    if (Node == nullptr) {
      TerminatorInst *T = Call->getParent()->getTerminator();
      revng_assert(not isTerminatorWithInvalidTarget(T, RootToIsolated));

      // Do nothing, don't emit the call to `function_call` and proceed
      return false;
    }

    auto *NameMD = cast<MDString>(&*Node->getOperand(0));
    TargetFunction = Functions.at(NameMD).IsolatedFunction;
  }

  revng_assert(TargetFunction != nullptr);

  // Create a builder object
  IRBuilder<> Builder(Context);
  Builder.SetInsertPoint(NewBB);

  CallInst *NewCall = nullptr;
  if (IsIndirect)
    NewCall = Builder.CreateCall(TargetFunction, { Call->getOperand(4) });
  else
    NewCall = Builder.CreateCall(TargetFunction);

  // Copy the func.call metadata from the terminator of the original block
  auto *T = Call->getParent()->getTerminator();
  MDNode *FuncCall = T->getMetadata("func.call");
  if (FuncCall != nullptr)
    NewCall->setMetadata("func.call", FuncCall);

  // Retrieve the fallthrough basic block and emit the branch
  BlockAddress *FallThroughAddress = cast<BlockAddress>(Call->getOperand(1));
  BasicBlock *FallthroughOld = FallThroughAddress->getBasicBlock();

  auto FallthroughOldIt = RootToIsolated.find(FallthroughOld);
  if (FallthroughOldIt != RootToIsolated.end()) {
    BasicBlock *FallthroughNew = cast<BasicBlock>(FallthroughOldIt->second);

    // Additional check for the return address PC
    LoadInst *ProgramCounter = Builder.CreateLoad(PC, "");
    ConstantInt *ExpectedPC = cast<ConstantInt>(Call->getOperand(2));
    Value *Result = Builder.CreateICmpEQ(ProgramCounter, ExpectedPC);

    // Create a basic block that we hit if the current PC is not the one
    // expected after the function call
    auto *PCMismatch = BasicBlock::Create(Context,
                                          NewBB->getName() + "_bad_return_pc",
                                          NewBB->getParent());
    throwException(BadReturnAddress, PCMismatch, ExpectedPC->getZExtValue());

    // Conditional branch to jump to the right block
    Builder.CreateCondBr(Result, FallthroughNew, PCMismatch);
  } else {

    // If the fallthrough basic block is not in the current function raise an
    // exception
    throwException(StandardTranslatedBlock, NewBB, 0);
  }

  return true;
}

bool IFI::isTerminatorWithInvalidTarget(Instruction *I,
                                        const ValueToValueMap &RootToIsolated) {
  if (auto *Terminator = dyn_cast<TerminatorInst>(I)) {

    // Here we check if among the successors of a terminator instruction
    // there is one that doesn't belong to the current function.
    for (BasicBlock *Target : Terminator->successors()) {
      if (RootToIsolated.count(Target) == 0) {
        return true;
      }
    }
  }

  return false;
}

bool IFI::cloneInstruction(BasicBlock *NewBB,
                           Instruction *OldInstruction,
                           IsolatedFunctionDescriptor &Descriptor) {
  Value *PCReg = getModule(NewBB)->getGlobalVariable(GCBI.pcReg()->getName(),
                                                     true);
  revng_assert(PCReg != nullptr);
  ValueToValueMap &RootToIsolated = Descriptor.ValueMap;

  // Create a builder object
  IRBuilder<> Builder(Context);
  Builder.SetInsertPoint(NewBB);

  if (isa<TerminatorInst>(OldInstruction)) {
    auto Type = Descriptor.Members.at(NewBB);
    switch (Type) {
    case StackAnalysis::BranchType::InstructionLocalCFG:
    case StackAnalysis::BranchType::FunctionLocalCFG:
    case StackAnalysis::BranchType::HandledCall:
    case StackAnalysis::BranchType::IndirectCall:
    case StackAnalysis::BranchType::IndirectTailCall:
    case StackAnalysis::BranchType::LongJmp:
    case StackAnalysis::BranchType::Killer:
    case StackAnalysis::BranchType::Unreachable:
    case StackAnalysis::BranchType::FakeFunctionCall:
      // These are handled later on
      break;

    case StackAnalysis::BranchType::Return:
      Builder.CreateRetVoid();
      return false;

    case StackAnalysis::BranchType::FakeFunctionReturn:
      Builder.CreateBr(Descriptor.FakeReturnPaths.at(NewBB));
      break;

    case StackAnalysis::BranchType::FakeFunction:
    case StackAnalysis::BranchType::FunctionSummary:
    case StackAnalysis::BranchType::IndirectTailCallFunction:
    case StackAnalysis::BranchType::Invalid:
    case StackAnalysis::BranchType::NoReturnFunction:
    case StackAnalysis::BranchType::UnhandledCall:
      revng_abort();
    }
  }

  if (isTerminatorWithInvalidTarget(OldInstruction, RootToIsolated)) {

    // We are in presence of a terminator with a successor no more in
    // the current function, let's throw an exception

    // If there's more than one successor, create a "trampoline" basic block
    // for each of them (lazily, `Descriptor.Trampolines` caches them).
    //
    // A trampoline stores the address of the corresponding successor in PC
    // and then throws the exception.

    // Lazily create the store-in-pc-and-throw trampoline
    auto GetTrampoline = [this, &Descriptor, PCReg](BasicBlock *BB) {
      Instruction *T = BB->getTerminator();

      auto *Node = cast_or_null<MDTuple>(T->getMetadata("func.entry"));
      auto FunctionsIt = Functions.end();
      if (Node != nullptr) {
        auto *NameMD = cast<MDString>(&*Node->getOperand(0));
        FunctionsIt = Functions.find(NameMD);
      }

      auto It = Descriptor.Trampolines.find(BB);
      if (It != Descriptor.Trampolines.end()) {
        return It->second;
      } else {

        auto *Trampoline = BasicBlock::Create(Context,
                                              "",
                                              Descriptor.IsolatedFunction);

        BlockType Type = GCBI.getType(BB);
        if (Type == AnyPCBlock or Type == UnexpectedPCBlock
            or Type == DispatcherBlock) {
          // The target is not a translated block, let's try to go through the
          // function dispatcher and let it throw the exception if necessary
          auto *Null = ConstantPointerNull::get(Type::getInt8PtrTy(Context));
          CallInst::Create(FunctionDispatcher, { Null }, "", Trampoline);
          ReturnInst::Create(Context, Trampoline);
        } else if (FunctionsIt != Functions.end()) {
          // The target is a function entry point, we assume we're dealing with
          // a tail call
          CallInst::Create(FunctionsIt->second.IsolatedFunction,
                           "",
                           Trampoline);
          ReturnInst::Create(Context, Trampoline);
        } else {
          uint64_t PC = getBasicBlockPC(BB);
          revng_assert(PC != 0);
          auto *PCType = PCReg->getType()->getPointerElementType();
          new StoreInst(ConstantInt::get(PCType, PC), PCReg, Trampoline);
          throwException(StandardNonTranslatedBlock, Trampoline, 0);
        }

        Descriptor.Trampolines[BB] = Trampoline;
        return Trampoline;
      }
    };

    // TODO: maybe cloning and patching would have been more effective
    switch (OldInstruction->getOpcode()) {
    case Instruction::Br: {
      auto *Branch = cast<BranchInst>(OldInstruction);
      if (Branch->isConditional()) {
        auto *Condition = RootToIsolated[Branch->getCondition()];
        Builder.CreateCondBr(Condition,
                             GetTrampoline(Branch->getSuccessor(0)),
                             GetTrampoline(Branch->getSuccessor(1)));
      } else {
        Builder.CreateBr(GetTrampoline(Branch->getSuccessor(0)));
      }
    } break;

    case Instruction::Switch: {
      auto *Switch = cast<SwitchInst>(OldInstruction);
      auto *Condition = RootToIsolated[Switch->getCondition()];
      auto *DefaultCase = GetTrampoline(Switch->getDefaultDest());
      auto *NewSwitch = Builder.CreateSwitch(Condition,
                                             DefaultCase,
                                             Switch->getNumCases());

      for (SwitchInst::CaseHandle &Case : Switch->cases()) {
        NewSwitch->addCase(Case.getCaseValue(),
                           GetTrampoline(Case.getCaseSuccessor()));
      }

    } break;

    default:
      revng_abort();
    }

  } else if (isCallTo(OldInstruction, "function_call")) {

    TerminatorInst *Terminator = OldInstruction->getParent()->getTerminator();
    if (isTerminatorWithInvalidTarget(Terminator, RootToIsolated)) {
      // Function call handling
      CallInst *Call = cast<CallInst>(OldInstruction);
      bool Result = replaceFunctionCall(NewBB, Call, RootToIsolated);

      // We return true if we emitted a function call to signal that we ended
      // the inspection of the current basic block and that we should exit from
      // the loop over the instructions
      return Result;
    }

  } else {

    // Actual copy of the instructions if we aren't in any of the corner
    // cases handled by the if before
    Instruction *NewInstruction = OldInstruction->clone();

    // Queue initialization with the base operand, the instruction
    // herself
    std::queue<User *> UserQueue;
    UserQueue.push(NewInstruction);

    // "Recursive" visit of the queue
    while (!UserQueue.empty()) {
      User *CurrentUser = UserQueue.front();
      UserQueue.pop();

      for (Use &CurrentUse : CurrentUser->operands()) {
        auto *CurrentOperand = CurrentUse.get();

        // Manage a standard value for which we find replacement in the
        // ValueToValueMap
        auto ReplacementIt = RootToIsolated.find(CurrentOperand);
        if (ReplacementIt != RootToIsolated.end()) {
          CurrentUse.set(ReplacementIt->second);

        } else if (auto *Address = dyn_cast<BlockAddress>(CurrentOperand)) {
          // Manage a BlockAddress
          Function *OldFunction = Address->getFunction();
          BasicBlock *OldBlock = Address->getBasicBlock();
          Function *NewFunction = cast<Function>(RootToIsolated[OldFunction]);
          BasicBlock *NewBlock = cast<BasicBlock>(RootToIsolated[OldBlock]);
          BlockAddress *B = BlockAddress::get(NewFunction, NewBlock);

          CurrentUse.set(B);

        } else if (isa<BasicBlock>(CurrentOperand)) {
          // Assert if we encounter a basic block and we don't find a
          // reference in the ValueToValueMap
          revng_assert(RootToIsolated.count(CurrentOperand) != 0);
        } else if (!isa<Constant>(CurrentOperand)
                   and !isa<MetadataAsValue>(CurrentOperand)) {
          // Manage values that are themself users (recursive exploration
          // of the operands) taking care of avoiding to add operands of
          // constants
          auto *CurrentSubUser = cast<User>(CurrentOperand);
          if (CurrentSubUser->getNumOperands() >= 1) {
            UserQueue.push(CurrentSubUser);
          }
        }
      }
    }

    if (OldInstruction->hasName()) {
      NewInstruction->setName(OldInstruction->getName());
    }

    Builder.Insert(NewInstruction);
    RootToIsolated[OldInstruction] = NewInstruction;
  }

  return false;
}

StringRef IFI::getFunctionNameString(MDNode *Node) {
  auto *Tuple = cast<MDTuple>(Node);
  QuickMetadata QMD(Context);
  StringRef FunctionNameString = QMD.extract<StringRef>(Tuple, 0);
  return FunctionNameString;
}

void IFI::run() {
  using namespace StackAnalysis::BranchType;

  // This function includes all the passages that realize the function
  // isolation. In particular the main steps of the function are:
  //
  // 1. Initialization
  // 2. Exception handling mechanism iniatilization
  // 3. Alloca harvesting
  // 4. Function call harvesting
  // 5. Function creation
  // 6. Function population
  // 7. Function inspection
  // 8. Function skeleton construction
  // 9. Reverse post order instantiation
  // 10. Removal of dummy switches
  // 11. Dummy Entry block
  // 12. Alloca placement
  // 13. Basic blocks population
  // 14. Exception handling control flow instantiation
  // 15. Module verification

  // 1. Initialize all the needed data structures

  // Assert if we don't find @function_call, sign that the function boundaries
  // analysis hasn't been run on the translated binary
  Function *CallMarker = TheModule->getFunction("function_call");
  revng_assert(CallMarker != nullptr);

  // 2. Create the needed structure to handle the throw of an exception

  // Retrieve the global variable corresponding to the program counter
  PC = TheModule->getGlobalVariable("pc", true);

  // Create a new global variable used as a flag for signaling the raise of an
  // exception
  auto *BoolTy = IntegerType::get(Context, 1);
  auto *ConstantFalse = ConstantInt::get(BoolTy, 0);
  new GlobalVariable(*TheModule,
                     Type::getInt1Ty(Context),
                     false,
                     GlobalValue::ExternalLinkage,
                     ConstantFalse,
                     "ExceptionFlag");
  ExceptionFlag = TheModule->getGlobalVariable("ExceptionFlag");

  // Declare the _Unwind_RaiseException function that we will use as a throw
  auto *RaiseExceptionFT = FunctionType::get(Type::getVoidTy(Context), false);

  RaiseException = Function::Create(RaiseExceptionFT,
                                    Function::ExternalLinkage,
                                    "raise_exception_helper",
                                    TheModule);

  // Create the Arrayref necessary for the arguments of exception_warning
  auto *IntegerType = IntegerType::get(Context, PCBitSize);
  std::vector<Type *> ArgsType{
    Type::getInt32Ty(Context), IntegerType, IntegerType, IntegerType
  };

  // Declare the exception_warning function
  auto *DebugExceptionFT = FunctionType::get(Type::getVoidTy(Context),
                                             ArgsType,
                                             false);

  DebugException = Function::Create(DebugExceptionFT,
                                    Function::ExternalLinkage,
                                    "exception_warning",
                                    TheModule);

  // Instantiate the dispatcher function, that is called in occurence of an
  // indirect function call.
  auto *FT = FunctionType::get(Type::getVoidTy(Context),
                               { Type::getInt8PtrTy(Context) },
                               false);

  // Creation of the function
  FunctionDispatcher = Function::Create(FT,
                                        Function::InternalLinkage,
                                        "function_dispatcher",
                                        TheModule);

  // 3. Search for all the alloca instructions and place them in an helper data
  //    structure in order to copy them at the beginning of the function where
  //    they are used. The alloca initially are all placed in the entry block of
  //    the root function.
  std::map<BasicBlock *, std::vector<Instruction *>> UsedAllocas;
  for (Instruction &I : RootFunction->getEntryBlock()) {

    // If we encounter an alloca copy it in the data structure that contains
    // all the allocas that we need to copy in the new basic block
    if (AllocaInst *Alloca = dyn_cast<AllocaInst>(&I)) {
      std::set<BasicBlock *> FilteredUsers;
      for (User *U : Alloca->users()) {

        // Handle standard instructions
        Instruction *UserInstruction = cast<Instruction>(U);
        FilteredUsers.insert(UserInstruction->getParent());

        // Handle the case in which we have ConstantExpr casting an alloca to
        // something else
        if (Value *SkippedCast = skipCasts(UserInstruction)) {
          FilteredUsers.insert(cast<Instruction>(SkippedCast)->getParent());
        }
      }
      for (BasicBlock *Parent : FilteredUsers) {
        UsedAllocas[Parent].push_back(Alloca);
      }
    }
  }

  // 4. Search for all the users of @function_call and populate the
  //    AdditionalSucc structure in order to be able to identify all the
  //    successors of a basic block
  std::map<BasicBlock *, BasicBlock *> AdditionalSucc;
  for (User *U : CallMarker->users()) {
    if (CallInst *Call = dyn_cast<CallInst>(U)) {
      BlockAddress *Fallthrough = cast<BlockAddress>(Call->getOperand(1));

      // Add entry in the data structure used to populate the dummy switches
      AdditionalSucc[Call->getParent()] = Fallthrough->getBasicBlock();
    }
  }

  // 5. Creation of the new LLVM functions on the basis of what recovered by
  //    the function boundaries analysis.

  for (BasicBlock &BB : *RootFunction) {
    revng_assert(!BB.empty());

    TerminatorInst *Terminator = BB.getTerminator();
    if (MDNode *Node = Terminator->getMetadata("func.entry")) {
      auto *FunctionNameMD = cast<MDString>(&*Node->getOperand(0));

      StringRef FunctionNameString = getFunctionNameString(Node);

      // We obtain a FunctionType of a function that has no arguments
      auto *FT = FunctionType::get(Type::getVoidTy(Context), false);

      // Check if we already have an entry for a function with a certain name
      revng_assert(Functions.count(FunctionNameMD) == 0);

      // Actual creation of an empty instance of a function
      Function *Function = Function::Create(FT,
                                            Function::InternalLinkage,
                                            FunctionNameString,
                                            TheModule);
      Function->setMetadata("func.entry", Node);

      IsolatedFunctionDescriptor &Descriptor = Functions[FunctionNameMD];
      Descriptor.PC = getBasicBlockPC(&BB);
      Descriptor.IsolatedFunction = Function;

      // Update v2v map with an ad-hoc mapping between the root function and
      // the current function, useful for subsequent analysis
      Descriptor.ValueMap[RootFunction] = Function;
    }
  }

  // 6. Population of the LLVM functions with the basic blocks that belong to
  //    them, always on the basis of the function boundaries analysis
  ConstantInt *ZeroValue = ConstantInt::get(Type::getInt8Ty(Context), 0);
  QuickMetadata QMD(Context);
  for (BasicBlock &BB : *RootFunction) {
    revng_assert(!BB.empty());

    // We iterate over all the metadata that represent the functions a basic
    // block belongs to, and add the basic block in each function
    TerminatorInst *Terminator = BB.getTerminator();
    if (MDNode *Node = Terminator->getMetadata("func.member.of")) {
      auto *Tuple = cast<MDTuple>(Node);
      for (const MDOperand &Op : Tuple->operands()) {
        auto *FunctionMD = cast<MDTuple>(Op);
        auto *FirstOperand = QMD.extract<MDTuple *>(FunctionMD, 0);
        auto *FunctionNameMD = QMD.extract<MDString *>(FirstOperand, 0);
        IsolatedFunctionDescriptor &Descriptor = Functions.at(FunctionNameMD);
        Function *ParentFunction = Descriptor.IsolatedFunction;

        // We assert if we can't find the parent function of the basic block
        revng_assert(ParentFunction != nullptr);

        // Create a new empty BB in the new function, preserving the original
        // name. We need to take care that if we are examining a basic block
        // that is the entry point of a function we need to place it in the as
        // the first block of the function.
        BasicBlock *NewBB;
        MDNode *FuncEntry = Terminator->getMetadata("func.entry");
        if (FuncEntry != nullptr
            and cast<MDString>(&*FuncEntry->getOperand(0)) == FunctionNameMD
            and not ParentFunction->empty()) {
          NewBB = BasicBlock::Create(Context,
                                     BB.getName(),
                                     ParentFunction,
                                     &ParentFunction->getEntryBlock());
        } else {
          NewBB = BasicBlock::Create(Context,
                                     BB.getName(),
                                     ParentFunction,
                                     nullptr);
        }

        // Update v2v map with the mapping between basic blocks
        Descriptor.ValueMap[&BB] = NewBB;

        // Update the map that we will use later for filling the basic blocks
        // with instructions
        IsolatedToRootBB[NewBB] = &BB;

        auto MemberType = fromName(QMD.extract<StringRef>(FunctionMD, 1));
        Descriptor.Members[NewBB] = MemberType;
      }
    }
  }

  // 7. Analyze all the created functions and populate them
  for (auto &Pair : Functions) {

    IsolatedFunctionDescriptor &Descriptor = Pair.second;

    // We are iterating over a map, so we need to extract the element from the
    // pair
    Function *AnalyzedFunction = Descriptor.IsolatedFunction;

    ValueToValueMap &RootToIsolated = Descriptor.ValueMap;

    // 8. We populate the basic blocks that are empty with a dummy switch
    //    instruction that has the role of preserving the actual shape of the
    //    function control flow. This will be helpful in order to traverse the
    //    BBs in reverse post-order.
    BasicBlock *UnexpectedPC = createUnreachableBlock("unexpectedpc",
                                                      AnalyzedFunction);
    BasicBlock *AnyPC = createUnreachableBlock("anypc", AnalyzedFunction);

    BasicBlock *RootUnexepctedPC = GCBI.unexpectedPC();
    RootToIsolated[RootUnexepctedPC] = UnexpectedPC;
    IsolatedToRootBB[UnexpectedPC] = RootUnexepctedPC;

    BasicBlock *RootAnyPC = GCBI.anyPC();
    RootToIsolated[RootAnyPC] = AnyPC;
    IsolatedToRootBB[AnyPC] = RootAnyPC;

    for (BasicBlock &NewBB : *AnalyzedFunction) {

      if (&NewBB == AnyPC or &NewBB == UnexpectedPC)
        continue;

      // Create a builder object
      IRBuilder<> Builder(Context);
      Builder.SetInsertPoint(&NewBB);

      if (Descriptor.Members.at(&NewBB) == FakeFunctionReturn) {
        Builder.CreateUnreachable();
        continue;
      }

      BasicBlock *BB = IsolatedToRootBB[&NewBB];
      revng_assert(BB != nullptr);
      TerminatorInst *Terminator = BB->getTerminator();

      // Collect all the successors of a basic block and add them in a proper
      // data structure
      std::vector<BasicBlock *> Successors;
      for (BasicBlock *Successor : Terminator->successors()) {

        revng_assert(GCBI.isTranslated(Successor)
                     || GCBI.getType(Successor) == AnyPCBlock
                     || GCBI.getType(Successor) == UnexpectedPCBlock
                     || GCBI.getType(Successor) == DispatcherBlock);
        auto SuccessorIt = RootToIsolated.find(Successor);

        // We add a successor if it is not a revng block type and it is present
        // in the VMap. It may be that we don't find a reference for Successor
        // in the RootToIsolated in case the block it is no more in the current
        // function. This happens for example in case we have a function call,
        // the target block of the final branch will be the entry block of the
        // callee, that for sure will not be in the current function and
        // consequently in the RootToIsolated.
        if (GCBI.isTranslated(Successor)
            and SuccessorIt != RootToIsolated.end()) {
          Successors.push_back(cast<BasicBlock>(SuccessorIt->second));
        }
      }

      // Add also the basic block that is executed after a function
      // call, identified before (the fall through block)
      if (BasicBlock *Successor = AdditionalSucc[BB]) {
        auto SuccessorIt = RootToIsolated.find(Successor);

        // In some occasions we have that the fallthrough block a function_call
        // is a block that doesn't belong to the current function
        // TODO: when the new function boundary detection algorithm will be in
        //       place check if this situation still occours or if we can assert
        if (SuccessorIt != RootToIsolated.end()) {
          Successors.push_back(cast<BasicBlock>(SuccessorIt->second));
        }
      }

      // Handle the degenerate case in which we didn't identified successors
      revng_assert(NewBB.getTerminator() == nullptr);
      if (Successors.size() == 0) {
        Builder.CreateUnreachable();
      } else {

        // Create the default case of the switch statement in an ad-hoc manner
        SwitchInst *DummySwitch = Builder.CreateSwitch(ZeroValue,
                                                       Successors.front());

        // Handle all the eventual successors except for the one already used
        // in the default case
        for (unsigned I = 1; I < Successors.size(); I++) {
          ConstantInt *Label = Builder.getInt8(I);
          DummySwitch->addCase(Label, Successors[I]);
        }
      }
    }

    //
    // Handle fake returns
    //

    // TODO: we do not support nested fake function calls
    // TODO: we do not support fake function calls sharing a fake return and,
    //       consequently, we do not support calling the same fake function
    //       multiple times
    for (auto &P : Descriptor.Members) {

      // Consider fake returns only
      if (P.second != StackAnalysis::BranchType::FakeFunctionReturn)
        continue;

      class FakeCallFinder : public BackwardBFSVisitor<FakeCallFinder> {
      public:
        FakeCallFinder(const IsolatedFunctionDescriptor &Descriptor) :
          Descriptor(Descriptor),
          FakeCall(nullptr) {}

        VisitAction visit(instruction_range Range) {
          revng_assert(Range.begin() != Range.end());
          auto *T = cast<TerminatorInst>(&*Range.begin());
          using namespace StackAnalysis::BranchType;
          if (Descriptor.Members.at(T->getParent()) == FakeFunctionCall) {
            revng_assert(FakeCall == nullptr or FakeCall == T->getParent(),
                         "Multiple fake function call sharing a fake return");
            FakeCall = T->getParent();
            return NoSuccessors;
          }

          return Continue;
        }

        BasicBlock *fakeCall() const { return FakeCall; }

      private:
        const IsolatedFunctionDescriptor &Descriptor;
        BasicBlock *FakeCall;
      };

      // Find the only corresponding fake function call
      FakeCallFinder FCF(Descriptor);
      FCF.run(P.first->getTerminator());
      BasicBlock *FakeCall = FCF.fakeCall();
      revng_assert(FakeCall != nullptr);

      // Get the fallthrough successor
      FakeCall = IsolatedToRootBB.at(FakeCall);
      Value *RootFallthrough = AdditionalSucc.at(FakeCall);
      auto *FakeFallthrough = cast<BasicBlock>(RootToIsolated[RootFallthrough]);

      // Replace unreachable with single-successor dummy switch
      TerminatorInst *T = P.first->getTerminator();
      revng_assert(isa<UnreachableInst>(T));
      T->eraseFromParent();
      SwitchInst::Create(ZeroValue, FakeFallthrough, 0, P.first);

      // Record in the descriptor for later usage
      Descriptor.FakeReturnPaths[P.first] = FakeFallthrough;
    }

    // 9. We instantiate the reverse post order on the skeleton we produced
    //    with the dummy switches
    revng_assert(not AnalyzedFunction->isDeclaration());
    ReversePostOrderTraversal<Function *> RPOT(AnalyzedFunction);

    // 10. We eliminate all the dummy switch instructions that we used before,
    //     and that should not appear in the output. The dummy switch are
    //     the first instruction of each basic block.
    for (BasicBlock &BB : *AnalyzedFunction) {

      // We exclude the unexpectedpc and anypc blocks since they have not been
      // populated with a dummy switch beforehand
      if (&BB != UnexpectedPC && &BB != AnyPC) {
        Instruction &I = *BB.begin();
        revng_assert(isa<SwitchInst>(I) || isa<UnreachableInst>(I));
        I.eraseFromParent();
      }
    }

    // 11. We add a dummy entry basic block that is useful for storing the
    //     alloca instructions used in each function and to avoid that the entry
    //     block has predecessors. The dummy entry basic blocks simply branches
    //     to the real entry block to have valid IR.
    BasicBlock *EntryBlock = &AnalyzedFunction->getEntryBlock();

    // Add a dummy block
    BasicBlock *Dummy = BasicBlock::Create(Context,
                                           "dummy_entry",
                                           AnalyzedFunction,
                                           EntryBlock);
    IRBuilder<> Builder(Dummy);

    // 12. We copy the allocas at the beginning of the function where they will
    //     be used
    std::set<Instruction *> AllocasToClone;
    for (BasicBlock &BB : *AnalyzedFunction)
      for (Instruction *OldAlloca : UsedAllocas[IsolatedToRootBB[&BB]])
        AllocasToClone.insert(OldAlloca);

    for (Instruction *OldAlloca : AllocasToClone) {
      Instruction *NewAlloca = OldAlloca->clone();
      if (OldAlloca->hasName()) {
        NewAlloca->setName(OldAlloca->getName());
      }

      // Please be aware that we are inserting the alloca after the dummy
      // switches, so until their removal done in phase 13 we will have
      // instruction after a terminator. This is done as we want to have the
      // dummy switches as first instructions in the basic blocks in order to
      // remove them by simply erasing the first instruction from each basic
      // block, instead of keeping track of them with an additional data
      // structure.
      Builder.Insert(NewAlloca);
      revng_assert(NewAlloca->getParent() == Dummy);
      RootToIsolated[OldAlloca] = NewAlloca;
    }

    // Create the unconditional branch to the real entry block
    BranchInst::Create(EntryBlock, Dummy);

    // 13. Visit of the basic blocks of the function in reverse post-order and
    //     population of them with the instructions
    for (BasicBlock *NewBB : RPOT) {

      // Do not try to populate unexpectedpc and anypc, since they have already
      // been populated in an ad-hoc manner.
      if (NewBB != UnexpectedPC && NewBB != AnyPC) {
        BasicBlock *OldBB = IsolatedToRootBB[NewBB];

        // Actual copy of the instructions
        for (Instruction &OldInstruction : *OldBB) {
          bool IsCall = cloneInstruction(NewBB, &OldInstruction, Descriptor);

          // If the cloneInstruction function returns true it means that we
          // emitted a function call and also the branch to the fallthrough
          // block, so we must end the inspection of the current basic block
          if (IsCall == true) {
            break;
          }
        }
      }
    }

    freeContainer(RootToIsolated);
  }

  // 14. Create the functions and basic blocks needed for the correct execution
  //     of the exception handling mechanism

  // Populate the function_dispatcher
  populateFunctionDispatcher();

  // Retrieve the root function, we use it a lot.
  Function *Root = TheModule->getFunction("root");

  // Get the unexpectedpc block of the root function
  // TODO: do this in a more elegant way (see if we have some helper)
  BasicBlock *UnexpectedPC = nullptr;
  for (BasicBlock &BB : *Root) {
    if (BB.getName() == "unexpectedpc")
      revng_assert(GCBI.getType(&BB) == UnexpectedPCBlock);
    if (GCBI.getType(&BB) == UnexpectedPCBlock) {
      UnexpectedPC = &BB;
      break;
    }
  }
  revng_assert(UnexpectedPC != nullptr);

  // Instantiate the basic block structure that handles the control flow after
  // an invoke
  BasicBlock *InvokeReturnBlock = createInvokeReturnBlock(Root, UnexpectedPC);

  // Instantiate the basic block structure that represents the catch of the
  // invoke, please remember that this is not used at the moment (exceptions
  // are handled in a customary way from the standard exit control flow path)
  BasicBlock *CatchBB = createCatchBlock(Root, UnexpectedPC);

  // Declaration of an ad-hoc personality function that is implemented in the
  // support.c source file
  auto *PersonalityFT = FunctionType::get(Type::getInt32Ty(Context), true);

  Function *PersonalityFunction = Function::Create(PersonalityFT,
                                                   Function::ExternalLinkage,
                                                   "exception_personality",
                                                   TheModule);

  // Add the personality to the root function
  Root->setPersonalityFn(PersonalityFunction);

  // Emit at the beginning of the basic blocks identified as function entries
  // by revng a call to the newly created corresponding LLVM function
  for (BasicBlock &BB : *Root) {
    revng_assert(!BB.empty());

    TerminatorInst *Terminator = BB.getTerminator();
    if (MDNode *Node = Terminator->getMetadata("func.entry")) {
      StringRef FunctionNameString = getFunctionNameString(Node);
      Function *TargetFunc = TheModule->getFunction(FunctionNameString);

      // Create a new trampoline entry block and substitute it to the old entry
      // block
      BasicBlock *NewBB = BasicBlock::Create(Context, "", BB.getParent(), &BB);
      BB.replaceAllUsesWith(NewBB);
      NewBB->takeName(&BB);

      // Emit the invoke instruction
      InvokeInst *Invoke = InvokeInst::Create(TargetFunc,
                                              InvokeReturnBlock,
                                              CatchBB,
                                              ArrayRef<Value *>(),
                                              "",
                                              NewBB);

      // Mark the invoke as non eligible for inlining that could break our
      // exception mechanism
      Invoke->setIsNoInline();
    }
  }

  // Remove all the orphan basic blocks from the root function (e.g., the
  // blocks that have been substitued by the trampoline)
  removeUnreachableBlocks(*Root);

  // 15. Before emitting it in output we check that the module in passes the
  //     verifyModule pass
  raw_os_ostream Stream(dbg);
  revng_assert(verifyModule(*TheModule, &Stream) == false);
}

bool IF::runOnModule(Module &TheModule) {

  // Retrieve analysis of the GeneratedCodeBasicInfo pass
  auto &GCBI = getAnalysis<GeneratedCodeBasicInfo>();

  // Create an object of type IsolateFunctionsImpl and run the pass
  IFI Impl(TheModule.getFunction("root"), GCBI);
  Impl.run();

  return false;
}
