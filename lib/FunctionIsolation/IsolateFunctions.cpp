/// \file IsolateFunctions.cpp
/// \brief Implements the IsolateFunctions pass which applies function isolation
///        using the informations provided by FunctionBoundariesDetectionPass.

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/raw_os_ostream.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/Local.h"

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
    MetaAddress PC;
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
  void throwException(Reason Code, BasicBlock *BB, MetaAddress AdditionalPC);

  /// \brief Instantiate a basic block that consists only of an exception throw
  BasicBlock *createUnreachableBlock(StringRef Name, Function *CurrentFunction);

  /// \brief Populate the @function_dispatcher, needed to handle the indirect
  ///        function calls
  void populateFunctionDispatcher();

  /// \brief Create the basic blocks that are hit on exit after an invoke
  ///        instruction
  BasicBlock *createInvokeReturnBlock(Function *Root, BasicBlock *Dispatcher);

  /// \brief Create the basic blocks that represent the catch of the invoke
  ///        instruction
  BasicBlock *createCatchBlock(Function *Root, BasicBlock *UnexpectedPC);

  // TODO: Make a class CloneHelper holding RootToIsolated as a member
  /// \brief Replace the calls marked by `func.call` with the actual call
  void replaceFunctionCall(StackAnalysis::BranchType::Values BranchType,
                           BasicBlock *NewBB,
                           Instruction *Call,
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
  Function *FunctionDispatcher;
  std::map<MDString *, IsolatedFunctionDescriptor> Functions;
  std::map<BasicBlock *, BasicBlock *> IsolatedToRootBB;
  GlobalVariable *PC;
  const unsigned PCBitSize;
};

void IFI::throwException(Reason Code,
                         BasicBlock *BB,
                         MetaAddress AdditionalPC) {
  revng_assert(PC != nullptr);
  revng_assert(RaiseException != nullptr);

  // Create a builder object
  IRBuilder<> Builder(Context);
  Builder.SetInsertPoint(BB);

  // Call the _debug_exception function to print usefull stuff
  LoadInst *ProgramCounter = Builder.CreateLoad(PC, "");

  MetaAddress LastPC;

  if (Code == StandardTranslatedBlock) {
    // Retrieve the value of the PC in the basic block where the exception has
    // been raised, this is possible since BB should be a translated block
    LastPC = getPC(&*BB->rbegin()).first;
    revng_assert(LastPC.isValid());
  } else {

    // The current basic block has not been translated from the original binary
    // (e.g. unexpectedpc or anypc), therefore we can't retrieve the
    // corresponding PC.
    LastPC = MetaAddress::invalid();
  }

  // Get the PC register dimension and use it to instantiate the arguments of
  // the call to exception_warning
  auto *ReasonValue = Builder.getInt32(Code);
  auto *ConstantLastPC = Builder.getIntN(PCBitSize, LastPC.asPCOrZero());
  auto *ConstantAdditionalPC = Builder.getIntN(PCBitSize,
                                               AdditionalPC.asPCOrZero());

  // Emit the call to the exception helper in support.c, which in turn calls the
  // exception_warning function and then the _Unwind_RaiseException
  Builder.CreateCall(RaiseException,
                     { ReasonValue,
                       ConstantLastPC,
                       ProgramCounter,
                       ConstantAdditionalPC });
  Builder.CreateUnreachable();
}

BasicBlock *
IFI::createUnreachableBlock(StringRef Name, Function *CurrentFunction) {

  // Create the basic block and add it in the function passed as parameter
  BasicBlock *NewBB = BasicBlock::Create(Context,
                                         Name,
                                         CurrentFunction,
                                         nullptr);

  throwException(StandardNonTranslatedBlock, NewBB, MetaAddress::invalid());
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
  throwException(FunctionDispatcherFallBack,
                 UnexpectedPC,
                 MetaAddress::invalid());
  setBlockType(UnexpectedPC->getTerminator(), BlockType::UnexpectedPCBlock);

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

    auto *Label = Builder.getIntN(PCBitSize, Descriptor.PC.asPC());
    Switch->addCase(Label, TrampolineBB);
  }
}

BasicBlock *
IFI::createInvokeReturnBlock(Function *Root, BasicBlock *Dispatcher) {

  // Create the first block
  BasicBlock *InvokeReturnBlock = BasicBlock::Create(Context,
                                                     "invoke_return",
                                                     Root,
                                                     nullptr);

  BranchInst::Create(Dispatcher, InvokeReturnBlock);

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

  Builder.CreateBr(UnexpectedPC);

  return CatchBB;
}

void IFI::replaceFunctionCall(StackAnalysis::BranchType::Values BranchType,
                              BasicBlock *NewBB,
                              Instruction *Call,
                              const ValueToValueMap &RootToIsolated) {
  //
  // Identify callee
  //
  BasicBlock *Callee = nullptr;
  if (auto *Branch = dyn_cast<BranchInst>(Call)) {
    if (Branch->isUnconditional()) {
      Callee = Branch->getSuccessor(0);
      if (GCBI.isPartOfRootDispatcher(Callee))
        Callee = nullptr;
    }
  }

  bool IsIndirect = (Callee == nullptr);

  //
  // Inspect `function_call` marker info
  //
  MetaAddress FallthroughPC = MetaAddress::invalid();
  BasicBlock *FallthroughOld = nullptr;
  Constant *FallthroughPCCI = nullptr;
  Value *ExternalFunctionName = nullptr;
  if (CallInst *FunctionCallMarker = getFunctionCall(Call)) {
    // Ensure we have the expected callee
    BasicBlock *MarkerCallee = nullptr;
    auto *FirstOperand = FunctionCallMarker->getArgOperand(0);
    auto *SecondOperand = FunctionCallMarker->getArgOperand(1);
    auto *ThirdOperand = FunctionCallMarker->getArgOperand(2);
    if (BlockAddress *MarkerCalleeBA = dyn_cast<BlockAddress>(FirstOperand)) {
      MarkerCallee = MarkerCalleeBA->getBasicBlock();
    }
    revng_assert(MarkerCallee == Callee);

    // Extract information about fallthrough basic block
    BlockAddress *FallThroughAddress = cast<BlockAddress>(SecondOperand);
    FallthroughOld = FallThroughAddress->getBasicBlock();
    FallthroughPC = MetaAddress::fromConstant(ThirdOperand);
    Type *PCType = PC->getType()->getPointerElementType();
    FallthroughPCCI = ConstantInt::get(PCType, FallthroughPC.asPC());

    // Extract external function callee name
    ExternalFunctionName = FunctionCallMarker->getOperand(4);
  } else {
    auto *FT = FunctionDispatcher->getType()->getPointerElementType();
    auto *Ptr = cast<PointerType>(cast<FunctionType>(FT)->getParamType(0));
    ExternalFunctionName = ConstantPointerNull::get(Ptr);
  }

  //
  // Identify callee's isolated function and its type
  //
  Function *TargetFunction = nullptr;
  bool IsNoReturn = false;
  if (IsIndirect) {
    TargetFunction = FunctionDispatcher;
  } else {
    Instruction *Terminator = Callee->getTerminator();
    auto *Node = cast<MDTuple>(Terminator->getMetadata("revng.func.entry"));
    auto *NameMD = cast<MDString>(&*Node->getOperand(0));
    IsolatedFunctionDescriptor &TargetDescriptor = Functions.at(NameMD);

    // Callee's llvm::Function
    TargetFunction = TargetDescriptor.IsolatedFunction;

    // Check the type
    auto *TypeMD = cast<MDString>(&*Node->getOperand(2));
    using namespace StackAnalysis::FunctionType;
    auto Type = fromName(TypeMD->getString());
    switch (Type) {
    case Regular:
      // Do nothing
      break;
    case NoReturn:
      IsNoReturn = true;
      break;
    case Invalid:
    case Fake:
    default:
      revng_abort();
    }
  }
  revng_assert(TargetFunction != nullptr);

  //
  // Coherency checks against the BranchType
  //
  switch (BranchType) {
  case StackAnalysis::BranchType::HandledCall:
    revng_assert(not IsIndirect);
    break;
  case StackAnalysis::BranchType::IndirectCall:
    revng_assert(IsIndirect);
    break;
  case StackAnalysis::BranchType::IndirectTailCall:
    revng_assert(IsIndirect);
    break;
  case StackAnalysis::BranchType::Killer:
  case StackAnalysis::BranchType::LongJmp:
    IsNoReturn = true;
    break;
  default:
    revng_abort();
  }

  //
  // Emit the code
  //

  // Create a builder object
  IRBuilder<> Builder(Context);
  Builder.SetInsertPoint(NewBB);

  // Emit the function call
  CallInst *NewCall = nullptr;
  if (IsIndirect)
    NewCall = Builder.CreateCall(TargetFunction, { ExternalFunctionName });
  else
    NewCall = Builder.CreateCall(TargetFunction);

  // Copy over the func.call metadata
  NewCall->setMetadata("func.call", Call->getMetadata("func.call"));

  //
  // Emit the branch to the return basic block, if necessary
  //
  if (IsNoReturn) {
    // If we return after a function call to a noreturn function, throw an
    // exception
    throwException(ReturnFromNoReturn, NewBB, MetaAddress::invalid());
  } else if (FallthroughOld) {
    // We have a fallthrough, emit a branch the fallthrough basic block
    auto FallthroughOldIt = RootToIsolated.find(FallthroughOld);
    if (FallthroughOldIt != RootToIsolated.end()) {
      BasicBlock *FallthroughNew = cast<BasicBlock>(FallthroughOldIt->second);

      // Additional check for the return address PC
      LoadInst *ProgramCounter = Builder.CreateLoad(PC, "");
      Value *Result = Builder.CreateICmpEQ(ProgramCounter, FallthroughPCCI);

      // Create a basic block that we hit if the current PC is not the one
      // expected after the function call
      auto *PCMismatch = BasicBlock::Create(Context,
                                            NewBB->getName() + "_bad_return_pc",
                                            NewBB->getParent());
      throwException(BadReturnAddress, PCMismatch, FallthroughPC);

      // Conditional branch to jump to the right block
      Builder.CreateCondBr(Result, FallthroughNew, PCMismatch);
    } else {
      // If the fallthrough basic block is not in the current function raise an
      // exception
      throwException(StandardTranslatedBlock, NewBB, MetaAddress::invalid());
    }
  } else {
    // Regular function call but now fallthrough basic block, it's a tail call
    Builder.CreateRetVoid();
  }
}

bool IFI::isTerminatorWithInvalidTarget(Instruction *I,
                                        const ValueToValueMap &RootToIsolated) {
  if (I->isTerminator()) {

    // Here we check if among the successors of a terminator instruction
    // there is one that doesn't belong to the current function.
    for (BasicBlock *Target : successors(I))
      if (RootToIsolated.count(Target) == 0)
        return true;
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
  bool IsCall = false;
  auto BranchType = StackAnalysis::BranchType::Invalid;

  if (OldInstruction->isTerminator()) {

    BranchType = Descriptor.Members.at(NewBB);
    switch (BranchType) {
    case StackAnalysis::BranchType::InstructionLocalCFG:
    case StackAnalysis::BranchType::FunctionLocalCFG:
    case StackAnalysis::BranchType::Killer:
    case StackAnalysis::BranchType::Unreachable:
    case StackAnalysis::BranchType::FakeFunctionCall:
      // These are handled later on
      break;

    case StackAnalysis::BranchType::HandledCall:
    case StackAnalysis::BranchType::IndirectCall:
    case StackAnalysis::BranchType::IndirectTailCall:
      IsCall = true;
      break;

    case StackAnalysis::BranchType::LongJmp:
    case StackAnalysis::BranchType::BrokenReturn:
      break;

    case StackAnalysis::BranchType::Return:
      Builder.CreateRetVoid();
      return false;

    case StackAnalysis::BranchType::FakeFunctionReturn:
      Builder.CreateBr(Descriptor.FakeReturnPaths.at(NewBB));
      return false;

    case StackAnalysis::BranchType::FakeFunction:
    case StackAnalysis::BranchType::RegularFunction:
    case StackAnalysis::BranchType::Invalid:
    case StackAnalysis::BranchType::NoReturnFunction:
    case StackAnalysis::BranchType::UnhandledCall:
      revng_abort();
    }
  }

  //
  // Handle function calls
  //
  bool IsCallToHelper = isCallToHelper(OldInstruction);
  if (not IsCallToHelper) {
    if (auto *FuncCallMD = OldInstruction->getMetadata("func.call")) {
      revng_assert(OldInstruction->isTerminator());
      replaceFunctionCall(BranchType, NewBB, OldInstruction, RootToIsolated);
      return true;
    }
  }

  revng_assert(not IsCall);

  if (isTerminatorWithInvalidTarget(OldInstruction, RootToIsolated)) {
    // We are in presence of a terminator with a successor no more in
    // the current function, let's throw an exception

    // If there's more than one successor, create a "trampoline" basic block
    // for each of them (lazily, `Descriptor.Trampolines` caches them).
    //
    // A trampoline stores the address of the corresponding successor in PC
    // and then throws the exception.

    // Lazily create the store-in-pc-and-throw trampoline
    // TODO: make a method
    auto GetTrampoline = [this, &Descriptor, PCReg](BasicBlock *BB) {
      Instruction *T = BB->getTerminator();

      auto *Node = cast_or_null<MDTuple>(T->getMetadata("revng.func.entry"));
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

        BlockType::Values Type = GCBI.getType(BB);
        MetaAddress PC = getBasicBlockPC(BB);
        if (Type == BlockType::AnyPCBlock
            or Type == BlockType::UnexpectedPCBlock
            or Type == BlockType::RootDispatcherBlock) {
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
        } else if (PC.isInvalid()) {
          // We're trying to jump to a basic block not starting with newpc, emit
          // an unreachable
          // TODO: emit a warning
          Module *M = Trampoline->getParent()->getParent();
          new UnreachableInst(M->getContext(), Trampoline);
        } else {
          auto *PCType = PCReg->getType()->getPointerElementType();
          new StoreInst(ConstantInt::get(PCType, PC.asPC()), PCReg, Trampoline);
          throwException(StandardNonTranslatedBlock,
                         Trampoline,
                         MetaAddress::invalid());
        }

        Descriptor.Trampolines[BB] = Trampoline;
        return Trampoline;
      }
    };

    // TODO: maybe cloning and patching would have been more effective
    Instruction *NewT = nullptr;
    switch (OldInstruction->getOpcode()) {
    case Instruction::Br: {
      auto *Branch = cast<BranchInst>(OldInstruction);
      if (Branch->isConditional()) {
        auto *Condition = RootToIsolated[Branch->getCondition()];
        NewT = Builder.CreateCondBr(Condition,
                                    GetTrampoline(Branch->getSuccessor(0)),
                                    GetTrampoline(Branch->getSuccessor(1)));
      } else {
        NewT = Builder.CreateBr(GetTrampoline(Branch->getSuccessor(0)));
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

      NewT = NewSwitch;

    } break;

    default:
      revng_abort();
    }

    if (auto *FuncCallMD = OldInstruction->getMetadata("func.call"))
      NewT->setMetadata("func.call", FuncCallMD);

  } else if (isCallTo(OldInstruction, "function_call")) {
    // Purge
  } else {

    // Actual copy of the instructions if we aren't in any of the corner
    // cases handled by the if before
    Instruction *NewInstruction = OldInstruction->clone();

    // Handle PHINodes
    if (auto *PHI = dyn_cast<PHINode>(NewInstruction)) {
      for (unsigned I = 0; I < PHI->getNumIncomingValues(); ++I) {
        auto *OldBB = PHI->getIncomingBlock(I);
        PHI->setIncomingBlock(I, cast<BasicBlock>(RootToIsolated[OldBB]));
      }
    }

    // Queue initialization with the base operand, the instruction
    // herself
    std::queue<Use *> UseQueue;
    for (Use &CurrentUse : NewInstruction->operands())
      UseQueue.push(&CurrentUse);

    // "Recursive" visit of the queue
    while (not UseQueue.empty()) {
      Use *CurrentUse = UseQueue.front();
      UseQueue.pop();

      auto *CurrentOperand = CurrentUse->get();

      // Manage a standard value for which we find replacement in the
      // ValueToValueMap
      auto ReplacementIt = RootToIsolated.find(CurrentOperand);
      if (ReplacementIt != RootToIsolated.end()) {
        revng_assert(not isa<Constant>(CurrentOperand));
        CurrentUse->set(ReplacementIt->second);
      } else if (isa<ConstantInt>(CurrentOperand)
                 or isa<GlobalObject>(CurrentOperand)
                 or isa<UndefValue>(CurrentOperand)
                 or isa<ConstantPointerNull>(CurrentOperand)) {
        // Do nothing
      } else if (auto *CE = dyn_cast<ConstantExpr>(CurrentOperand)) {
        for (Use &U : CE->operands())
          UseQueue.push(&U);
      } else if (auto *CA = dyn_cast<ConstantAggregate>(CurrentOperand)) {
        for (Use &U : CA->operands())
          UseQueue.push(&U);
      } else {
        revng_abort();
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

  // Create the Arrayref necessary for the arguments of raise_exception_helper
  auto *IntegerType = IntegerType::get(Context, PCBitSize);
  std::vector<Type *> ArgsType{
    Type::getInt32Ty(Context), IntegerType, IntegerType, IntegerType
  };

  // Declare the _Unwind_RaiseException function that we will use as a throw
  auto *RaiseExceptionFT = FunctionType::get(Type::getVoidTy(Context),
                                             ArgsType,
                                             false);

  RaiseException = Function::Create(RaiseExceptionFT,
                                    Function::ExternalLinkage,
                                    "raise_exception_helper",
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

    Instruction *Terminator = BB.getTerminator();
    if (MDNode *Node = Terminator->getMetadata("revng.func.entry")) {
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
      Function->setMetadata("revng.func.entry", Node);

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
    Instruction *Terminator = BB.getTerminator();
    if (MDNode *Node = Terminator->getMetadata("revng.func.member.of")) {
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
        MDNode *FuncEntry = Terminator->getMetadata("revng.func.entry");
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
    setBlockType(UnexpectedPC->getTerminator(), BlockType::UnexpectedPCBlock);
    BasicBlock *AnyPC = createUnreachableBlock("anypc", AnalyzedFunction);
    setBlockType(AnyPC->getTerminator(), BlockType::AnyPCBlock);

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
      Instruction *Terminator = BB->getTerminator();

      // Collect all the successors of a basic block and add them in a proper
      // data structure
      std::vector<BasicBlock *> Successors;
      for (BasicBlock *Successor : successors(Terminator)) {

        auto SuccessorType = GCBI.getType(Successor);
        switch (SuccessorType) {
        case BlockType::AnyPCBlock:
        case BlockType::UnexpectedPCBlock:
        case BlockType::RootDispatcherBlock:
        case BlockType::IndirectBranchDispatcherHelperBlock:
          break;
        default:
          revng_assert(GCBI.isTranslated(Successor));
          break;
        }
        auto SuccessorIt = RootToIsolated.find(Successor);

        // We add a successor if it is not a revng block type and it is present
        // in the VMap. It may be that we don't find a reference for Successor
        // in the RootToIsolated in case the block it is no more in the current
        // function. This happens for example in case we have a function call,
        // the target block of the final branch will be the entry block of the
        // callee, that for sure will not be in the current function and
        // consequently in the RootToIsolated.
        if ((GCBI.isTranslated(Successor)
             or SuccessorType == BlockType::IndirectBranchDispatcherHelperBlock)
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
          ConstantInt *Label = Builder.getInt8(static_cast<uint8_t>(I));
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
          Descriptor(Descriptor), FakeCall(nullptr) {}

        VisitAction visit(instruction_range Range) {
          revng_assert(Range.begin() != Range.end());
          Instruction *Term = &*Range.begin();
          using namespace StackAnalysis::BranchType;
          if (Descriptor.Members.at(Term->getParent()) == FakeFunctionCall) {
            revng_assert(FakeCall == nullptr or FakeCall == Term->getParent(),
                         "Multiple fake function call sharing a fake return");
            FakeCall = Term->getParent();
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
      Instruction *T = P.first->getTerminator();
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
        BasicBlock *OldBB = IsolatedToRootBB.at(NewBB);
        revng_assert(OldBB != nullptr);

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

        {
          using namespace StackAnalysis::BranchType;
          Values Type = Descriptor.Members.at(NewBB);
          Instruction *T = NewBB->getTerminator();
          T->setMetadata("member.type", QMD.tuple(getName(Type)));
        }
      }

      unsigned Terminators = 0;
      for (Instruction &I : *NewBB)
        if (I.isTerminator())
          Terminators++;
      revng_assert(Terminators == 1);
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
  BasicBlock *UnexpectedPC = GCBI.unexpectedPC();
  BasicBlock *Dispatcher = GCBI.dispatcher();
  revng_assert(UnexpectedPC != nullptr);
  revng_assert(Dispatcher != nullptr);

  // Instantiate the basic block structure that handles the control flow after
  // an invoke
  BasicBlock *InvokeReturnBlock = createInvokeReturnBlock(Root, Dispatcher);

  // Instantiate the basic block structure that represents the catch of the
  // invoke, please remember that this is not used at the moment (exceptions
  // are handled in a customary way from the standard exit control flow path)
  BasicBlock *CatchBB = createCatchBlock(Root, UnexpectedPC);

  // Declaration of an ad-hoc personality function that is implemented in the
  // support.c source file
  auto *PersonalityFT = FunctionType::get(Type::getInt32Ty(Context), true);

  Function *PersonalityFunction = Function::Create(PersonalityFT,
                                                   Function::ExternalLinkage,
                                                   "__gxx_personality_v0",
                                                   TheModule);

  // Add the personality to the root function
  Root->setPersonalityFn(PersonalityFunction);

  // Emit at the beginning of the basic blocks identified as function entries
  // by revng a call to the newly created corresponding LLVM function
  for (BasicBlock &BB : *Root) {
    revng_assert(!BB.empty());

    Instruction *Terminator = BB.getTerminator();
    if (MDNode *Node = Terminator->getMetadata("revng.func.entry")) {
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

  // Remove all the orphan basic blocks from the root function (e.g., the blocks
  // that have been substitued by the trampoline)
  {
    ReversePostOrderTraversal<BasicBlock *> RPOT(&Root->getEntryBlock());
    std::set<BasicBlock *> Reachable;
    for (BasicBlock *BB : RPOT)
      Reachable.insert(BB);

    std::vector<BasicBlock *> ToDelete;
    for (BasicBlock &BB : *Root)
      if (Reachable.count(&BB) == 0)
        ToDelete.push_back(&BB);

    for (BasicBlock *BB : ToDelete)
      BB->dropAllReferences();

    for (BasicBlock *BB : ToDelete)
      BB->eraseFromParent();
  }

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
