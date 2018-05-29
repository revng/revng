/// \file isolatefunctions.cpp
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

// Local includes
#include "commonconstants.h"
#include "debug.h"
#include "generatedcodebasicinfo.h"
#include "ir-helpers.h"
#include "isolatefunctions.h"

using namespace llvm;

class IsolateFunctionsImpl;

// Define an alias for the data structure that will contain the LLVM functions
using FunctionsMap = std::map<MDString *, Function *>;

typedef DenseMap<const Value*, Value*> ValueToValueMap;

using IF = IsolateFunctions;
using IFI = IsolateFunctionsImpl;

char IF::ID = 0;
static RegisterPass<IF> X("if", "Isolate Functions Pass", true, true);

class IsolateFunctionsImpl {
public:
  IsolateFunctionsImpl(Function &RootFunction,
                       Module *NewModule,
                       GeneratedCodeBasicInfo &GCBI,
                       ValueToValueMapTy &ModuleCloningVMap) :
  RootFunction(RootFunction),
  NewModule(NewModule),
  GCBI(GCBI),
  ModuleCloningVMap(ModuleCloningVMap),
  Context(getContext(NewModule)),
  PCBitSize(8 * GCBI.pcRegSize()) {
  }

  void run();

private:

  /// \brief Creates the call that simulates the throw of an exception
  void throwException(Reason Code, BasicBlock *BB, uint64_t AdditionalPC);

  /// \brief Instantiate a basic block that consists only of an exception throw
  BasicBlock *createUnreachableBlock(StringRef Name,
                                     Function *CurrentFunction);

  /// \brief Populate the @function_dispatcher, needed to handle the indirect
  ///        function calls
  void populateFunctionDispatcher();

  /// \brief Create the basic blocks that are hit on exit after an invoke
  ///        instruction
  BasicBlock *createInvokeReturnBlock(Function *Root,
                                      BasicBlock *UnexpectedPC);

  /// \brief Create the basic blocks that represent the catch of the invoke
  ///        instruction
  BasicBlock *createCatchBlock(Function *Root,
                               BasicBlock *UnexpectedPC);

  /// \brief Replace the call to the @function_call marker with the actual call
  void replaceFunctionCall(BasicBlock *NewBB,
                           CallInst *Call,
                           const ValueToValueMap &LocalVMap);

  /// \brief Checks if an instruction is a terminator with an invalid successor
  bool isTerminatorWithInvalidTarget(Instruction *I,
                                     const ValueToValueMap &LocalVMap);

  /// \brief Handle the cloning of an instruction in the new basic block
  ///
  /// \return true if the function purged all the instructions after this one in
  ///         the current basic block
  bool cloneInstruction(BasicBlock *NewBB,
                        Instruction *OldInstruction,
                        ValueToValueMap &LocalVMap);

  /// \brief Extract the string representing a function name starting from the
  ///        MDNode
  /// \return StringRef representing the function name
  StringRef getFunctionNameString(MDNode *Node);

private:
  Function &RootFunction;
  Module *NewModule;
  GeneratedCodeBasicInfo &GCBI;
  ValueToValueMapTy &ModuleCloningVMap;
  LLVMContext &Context;
  Function *RaiseException;
  Function *DebugException;
  Function *FunctionDispatcher;
  std::map<BasicBlock *, BasicBlock *> NewToOldBBMap;
  std::map<Function *, uint64_t> FunctionsPC;
  GlobalVariable *ExceptionFlag;
  GlobalVariable *PC;
  const unsigned PCBitSize;
};

void IFI::throwException(Reason Code, BasicBlock *BB, uint64_t AdditionalPC) {
  assert(PC != nullptr);
  assert(RaiseException != nullptr);
  assert(DebugException != nullptr);

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
    assert(LastPC != 0);
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
                     {
                       ReasonValue,
                       ConstantLastPC,
                       ProgramCounter,
                       ConstantAdditionalPC
                     },
                     "");

  // Emit the call to _Unwind_RaiseException
  Builder.CreateCall(RaiseException);
}

BasicBlock *IFI::createUnreachableBlock(StringRef Name,
                                        Function *CurrentFunction) {

    // Create the basic block and add it in the function passed as parameter
    BasicBlock* NewBB = BasicBlock::Create(Context,
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
  new UnreachableInst(Context, UnexpectedPC);

  // Create a builder object for the DispatcherBB basic block
  IRBuilder<> Builder(Context);
  Builder.SetInsertPoint(DispatcherBB);

  LoadInst *ProgramCounter = Builder.CreateLoad(PC, "");

  SwitchInst *Switch = Builder.CreateSwitch(ProgramCounter, UnexpectedPC);

  for (auto &Pair : FunctionsPC) {
    Function *Function = Pair.first;
    StringRef Name = Function->getName();

    // Creation of a basic block correspondent to the trampoline for each
    // function
    BasicBlock *TrampolineBB = BasicBlock::Create(Context,
                                                  Name + "_trampoline",
                                                  FunctionDispatcher,
                                                  nullptr);

    CallInst::Create(Function, "", TrampolineBB);
    ReturnInst::Create(Context, TrampolineBB);

    uint64_t FunctionPC = Pair.second;
    auto *Label = Builder.getIntN(PCBitSize, FunctionPC);
    Switch->addCase(Label, TrampolineBB);
  }
}

BasicBlock *IFI::createInvokeReturnBlock(Function *Root,
                                         BasicBlock *UnexpectedPC) {

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

BasicBlock *IFI::createCatchBlock(Function *Root,
                                  BasicBlock *UnexpectedPC) {

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
  std::vector<Type *> InArgsType { RetTyPointerType, RetTyIntegerType };
  StructType *RetTyStruct = StructType::create(Context,
                                               ArrayRef<Type *>(InArgsType),
                                               "",
                                               false);

  // Create the landingpad instruction
  LandingPadInst *LandingPad = Builder.CreateLandingPad(RetTyStruct, 0);

  // Add a catch all (constructed with the null value as clause)
  LandingPad->addClause(ConstantPointerNull::get(Type::getInt8PtrTy(Context)));
  Builder.CreateUnreachable();

  return CatchBB;
}

void IFI::replaceFunctionCall(BasicBlock *NewBB,
                              CallInst *Call,
                              const ValueToValueMap &LocalVMap) {

  // Retrieve the called function and emit the call
  StringRef FunctionNameString;

  if (BlockAddress *Callee = dyn_cast<BlockAddress>(Call->getOperand(0))){
    BasicBlock *CalleeEntry = Callee->getBasicBlock();
    TerminatorInst *Terminator = CalleeEntry->getTerminator();
    MDNode *Node = Terminator->getMetadata("func.entry");
    FunctionNameString = getFunctionNameString(Node);
  } else {
    FunctionNameString = FunctionDispatcher->getName();
  }

  Function *TargetFunction = NewModule->getFunction(FunctionNameString);
  assert(TargetFunction != nullptr);

  // Create a builder object
  IRBuilder<> Builder(Context);
  Builder.SetInsertPoint(NewBB);

  Builder.CreateCall(TargetFunction);

  // Retrieve the fallthrough basic block and emit the branch
  BlockAddress *FallThroughAddress = cast<BlockAddress>(Call->getOperand(1));
  BasicBlock *FallthroughOld = FallThroughAddress->getBasicBlock();

  auto FallthroughOldIt = LocalVMap.find(FallthroughOld);
  if (FallthroughOldIt != LocalVMap.end()) {
    BasicBlock *FallthroughNew = cast<BasicBlock>(FallthroughOldIt->second);

    // Additional check for the return address PC
    LoadInst *ProgramCounter = Builder.CreateLoad(PC, "");
    ConstantInt *ExpectedPC = cast<ConstantInt>(Call->getOperand(2));
    Value *Result = Builder.CreateICmpEQ(ProgramCounter, ExpectedPC);

    // Create a basic block that we hit if the current PC is not the one
    // expected after the function call
    Twine PCMismatchName = NewBB->getName() + "_bad_return_pc";
    BasicBlock *PCMismatch = BasicBlock::Create(Context,
                                                PCMismatchName.str(),
                                                NewBB->getParent(),
                                                nullptr);
    throwException(BadReturnAddress, PCMismatch, ExpectedPC->getZExtValue());
    new UnreachableInst(Context, PCMismatch);

    // Conditional branch to jump to the right block
    Builder.CreateCondBr(Result, FallthroughNew, PCMismatch);
  } else {

    // If the fallthrough basic block is not in the current function raise an
    // exception
    throwException(StandardTranslatedBlock, NewBB, 0);
    Builder.CreateUnreachable();
  }
}

bool IFI::isTerminatorWithInvalidTarget(Instruction *I,
                                        const ValueToValueMap &LocalVMap) {
  if (auto *Terminator = dyn_cast<TerminatorInst>(I)) {

    // Here we check if among the successors of a terminator instruction
    // there is one that doesn't belong anymore to the current function.
    for (BasicBlock *Target : Terminator->successors()) {
      if (LocalVMap.count(Target) == 0) {
        return true;
      }
    }
  }

  return false;
}

bool IFI::cloneInstruction(BasicBlock *NewBB,
                           Instruction *OldInstruction,
                           ValueToValueMap &LocalVMap) {

  // Create a builder object
  IRBuilder<> Builder(Context);
  Builder.SetInsertPoint(NewBB);

  // Check if the function boundaries analysis has identified an instruction as
  // a ret and in that case emit a ret instruction
  if (OldInstruction->getMetadata("func.return") != nullptr) {
    Builder.CreateRetVoid();

  } else if (isTerminatorWithInvalidTarget(OldInstruction, LocalVMap)) {

    // If we are in presence of a terminator with a successor no more in the
    // current function we throw an exception
    throwException(StandardTranslatedBlock, NewBB, 0);
    Builder.CreateUnreachable();

  } else if (isCallTo(OldInstruction, "function_call")) {

    // Function call handling
    CallInst *Call = cast<CallInst>(OldInstruction);
    replaceFunctionCall(NewBB,
                        Call,
                        LocalVMap);

    // We return true if we emitted a function call to signal that we ended
    // the inspection of the current basic block and that we should exit from
    // the loop over the instructions
    return true;

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
        auto ReplacementIt = LocalVMap.find(CurrentOperand);
        if (ReplacementIt != LocalVMap.end()) {
          CurrentUse.set(ReplacementIt->second);

        } else if (auto *Address = dyn_cast<BlockAddress>(CurrentOperand)) {
          // Manage a BlockAddress
          Function *OldFunction = Address->getFunction();
          BasicBlock *OldBlock = Address->getBasicBlock();
          Function *NewFunction = cast<Function>(LocalVMap[OldFunction]);
          BasicBlock *NewBlock  = cast<BasicBlock>(LocalVMap[OldBlock]);
          BlockAddress *B = BlockAddress::get(NewFunction, NewBlock);

          CurrentUse.set(B);

        } else if (isa<BasicBlock>(CurrentOperand)) {
          // Assert if we encounter a basic block and we don't find a
          // reference in the ValueToValueMap
          assert(LocalVMap.count(CurrentOperand) != 0);
        } else if (!isa<Constant>(CurrentOperand)) {
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
    LocalVMap[OldInstruction] = NewInstruction;
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
  assert(RootFunction.getParent()->getFunction("function_call") != nullptr);
  Function *CallMarker = RootFunction.getParent()->getFunction("function_call");

  // Fill the GlobalVMap to contain the mappings made by the CloneModule
  // function, in order to have the mappings between global objects (global
  // variables and functions). We'll initialize the LocalVMaps of the single
  // functions with these mappings.
  ValueToValueMap GlobalVMap;
  for (auto Iter : ModuleCloningVMap) {
    if (isa<GlobalObject>(Iter.first)) {
      GlobalVMap[Iter.first] = Iter.second;
    }
  }

  // 2. Create the needed structure to handle the throw of an exception

  // Retrieve the global variable corresponding to the program counter
  PC = NewModule->getGlobalVariable("pc", true);

  // Create a new global variable used as a flag for signaling the raise of an
  // exception
  auto *BoolTy = IntegerType::get(Context, 1);
  auto *ConstantFalse = ConstantInt::get(BoolTy, 0);
  new GlobalVariable(*NewModule,
                     Type::getInt1Ty(Context),
                     false,
                     GlobalValue::ExternalLinkage,
                     ConstantFalse,
                     "ExceptionFlag");
  ExceptionFlag = NewModule->getGlobalVariable("ExceptionFlag");

  // Declare the _Unwind_RaiseException function that we will use as a throw
  auto *RaiseExceptionFT = FunctionType::get(Type::getVoidTy(Context), false);

  RaiseException = Function::Create(RaiseExceptionFT,
                                    Function::ExternalLinkage,
                                    "raise_exception_helper",
                                    NewModule);

  // Create the Arrayref necessary for the arguments of exception_warning
  auto *IntegerType = IntegerType::get(Context, PCBitSize);
  std::vector<Type *> ArgsType {
                                 Type::getInt32Ty(Context),
                                 IntegerType,
                                 IntegerType,
                                 IntegerType
                               };

  // Declare the exception_warning function
  auto *DebugExceptionFT = FunctionType::get(Type::getVoidTy(Context),
                                             ArgsType,
                                             false);

  DebugException = Function::Create(DebugExceptionFT,
                                    Function::ExternalLinkage,
                                    "exception_warning",
                                    NewModule);

  // Instantiate the dispatcher function, that is called in occurence of an
  // indirect function call.
  auto *FT = FunctionType::get(Type::getVoidTy(Context), false);

  // Creation of the function
  FunctionDispatcher = Function::Create(FT,
                                        Function::ExternalLinkage,
                                        "function_dispatcher",
                                        NewModule);

  // 3. Search for all the alloca instructions and place them in an helper data
  //    structure in order to copy them at the beginning of the function where
  //    they are used. The alloca initially are all placed in the entry block of
  //    the root function.
  std::map<BasicBlock *, std::vector<Instruction *>> UsedAllocas;
  for (Instruction &I : RootFunction.getEntryBlock()) {

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

  // 4. Search for all the users of the helper function @function_call and
  //    populate the AdditionalSucc structure in order to be able to identify
  //    all the successors of a basic block
  std::map<BasicBlock *, BasicBlock *> AdditionalSucc;
  for (User *U : CallMarker->users()) {
    if (CallInst *Call = dyn_cast<CallInst>(U)) {
      BlockAddress *Fallthrough = cast<BlockAddress>(Call->getOperand(1));

      // Add entry in the data structure used to populate the dummy switches
      AdditionalSucc[Call->getParent()] = Fallthrough->getBasicBlock();
    }
  }

  // 5. Creation of the new LLVM functions on the basis of what recovered by
  //    the function boundaries analysis and storage of the pointers in a
  //    dedicated data strucure. We also initialize each VMap contained in the
  //    MetaVMap structure with the mappings contained in GlobalVMap.
  std::map<MDString *, Function *> Functions;
  std::map<Function *, ValueToValueMap> MetaVMap;

  for (BasicBlock &BB : RootFunction) {
    assert(!BB.empty());

    TerminatorInst *Terminator = BB.getTerminator();
    if (MDNode *Node = Terminator->getMetadata("func.entry")) {
      auto *FunctionNameMD = cast<MDString>(&*Node->getOperand(0));

      StringRef FunctionNameString = getFunctionNameString(Node);

      // We obtain a FunctionType of a function that has no arguments
      auto *FT = FunctionType::get(Type::getVoidTy(Context), false);

      // Check if we already have an entry for a function with a certain name
      if (Functions.count(FunctionNameMD) == 0) {

        // Actual creation of an empty instance of a function
        Function *Function =  Function::Create(FT,
                                               Function::ExternalLinkage,
                                               FunctionNameString,
                                               NewModule);

        Functions[FunctionNameMD] = Function;
        FunctionsPC[Function] = getBasicBlockPC(&BB);

        // Update v2v map with an ad-hoc mapping between the root function and
        // the current function, useful for subsequent analysis
        ValueToValueMap &LocalVMap = MetaVMap[Function];

        // Copy all the mappings between global variables already created when
        // we cloned the module
        LocalVMap = GlobalVMap;

        // Add the mapping between root function and all the functions we
        // will create
        LocalVMap[&RootFunction] = Function;
      }
    }
  }

  // 6. Population of the LLVM functions with the basic blocks that belong to
  //    them, always on the basis of the function boundaries analysis
  for (BasicBlock &BB : RootFunction) {
    assert(!BB.empty());

    // We iterate over all the metadata that represent the functions a basic
    // block belongs to, and add the basic block in each function
    TerminatorInst *Terminator = BB.getTerminator();
    if (MDNode *Node = Terminator->getMetadata("func.member.of")) {
      auto *Tuple = cast<MDTuple>(Node);
      for (const MDOperand &Op : Tuple->operands()) {
        auto *FunctionMD = cast<MDTuple>(Op);
        auto *FunctionNameMD = cast<MDString>(&*FunctionMD->getOperand(0));
        Function *ParentFunction = Functions[FunctionNameMD];

        // We assert if we can't find the parent function of the basic block
        assert(ParentFunction != nullptr);

        // Creation of a new empty BB in the new generated corresponding
        // function, preserving the original name. We need to take care that if
        // we are examining a basic block that is the entry point of a function
        // we need to place it in the as the first block of the function.
        BasicBlock* NewBB;
        if (Terminator->getMetadata("func.entry") && !ParentFunction->empty()) {
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
        ValueToValueMap &LocalVMap = MetaVMap[ParentFunction];
        LocalVMap[&BB] = NewBB;

        // Update the map that we will use later for filling the basic blocks
        // with instructions
        NewToOldBBMap[NewBB] = &BB;
      }
    }
  }

  // 7. Analyze all the created functions and populate them
  for (auto &Pair : Functions) {

    // We are iterating over a map, so we need to extract the element from the
    // pair
    Function *AnalyzedFunction = Pair.second;

    // Initialize a local ValueToValueMap with the mapping between basic
    // blocks (done in the previous loop) and the global objects contained
    // in the correspondig VMap in the MetaVMap structure
    ValueToValueMap LocalVMap = std::move(MetaVMap[AnalyzedFunction]);

    // 8. We populate the basic blocks that are empty with a dummy switch
    //    instruction that has the role of preserving the actual shape of the
    //    function control flow. This will be helpful in order to traverse the
    //    BBs in reverse post-order.
    BasicBlock* UnexpectedPC = nullptr;
    BasicBlock* AnyPC = nullptr;

    for (BasicBlock &NewBB : *AnalyzedFunction) {

      BasicBlock *BB = NewToOldBBMap[&NewBB];
      TerminatorInst *Terminator = BB->getTerminator();

      // Collect all the successors of a basic block and add them in a proper
      // data structure
      std::vector<BasicBlock *> Successors;
      for (BasicBlock *Successor : Terminator->successors()) {

        // Check if among the successors of the current basic block there is
        // the unexpectedpc basic block, and if needed create it
        if (GCBI.getType(Successor) == UnexpectedPCBlock) {
          // Check if it already exists and create an unexpectedpc block
          if (UnexpectedPC == nullptr) {
            UnexpectedPC = createUnreachableBlock("unexpectedpc",
                                                  AnalyzedFunction);
            LocalVMap[Successor] = UnexpectedPC;
            NewToOldBBMap[UnexpectedPC] = Successor;
          }
        }

        // Check if among the successors of the current basic block there is
        // the anypc basic block, and if needed create it
        if (GCBI.getType(Successor) == AnyPCBlock) {
          // Check if it already exists and create an anypc block
          if (AnyPC == nullptr) {
            AnyPC = createUnreachableBlock("anypc",
                                           AnalyzedFunction);
            LocalVMap[Successor] = AnyPC;
            NewToOldBBMap[AnyPC] = Successor;
          }
        }

        assert(GCBI.isTranslated(Successor)
               || GCBI.getType(Successor) == AnyPCBlock
               || GCBI.getType(Successor) == UnexpectedPCBlock
               || GCBI.getType(Successor) == DispatcherBlock);
        auto SuccessorIt = LocalVMap.find(Successor);

        // We add a successor if it is not a revamb block type and it is present
        // in the VMap. It may be that we don't find a reference for Successor
        // in the LocalVMap in case the block it is no more in the current
        // function. This happens for example in case we have a function call,
        // the target block of the final branch will be the entry block of the
        // callee, that for sure will not be in the current function and
        // consequently in the LocalVMap.
        if (GCBI.isTranslated(Successor) && SuccessorIt != LocalVMap.end()) {
          Successors.push_back(cast<BasicBlock>(SuccessorIt->second));
        }
      }

      // Add also the basic block that is executed after a function
      // call, identified before (the fall through block)
      if (BasicBlock *Successor = AdditionalSucc[BB]) {
        auto SuccessorIt = LocalVMap.find(Successor);

        // In some occasions we have that the fallthrough block a function_call
        // is a block that doesn't belong to the current function
        // TODO: when the new function boundary detection algorithm will be in
        //       place check if this situation still occours or if we can assert
        if (SuccessorIt != LocalVMap.end()) {
          Successors.push_back(cast<BasicBlock>(SuccessorIt->second));
        }
      }

      // Create a builder object
      IRBuilder<> Builder(Context);
      Builder.SetInsertPoint(&NewBB);

      // Handle the degenerate case in which we didn't identified successors
      if(Successors.size() == 0) {
        Builder.CreateUnreachable();
      } else {

        // Create the default case of the switch statement in an ad-hoc manner
        ConstantInt *ZeroValue = Builder.getInt8(0);
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

    // 9. We instantiate the reverse post order on the skeleton we produced
    //    with the dummy switches
    ReversePostOrderTraversal<Function *> RPOT(AnalyzedFunction);

    // 10. We eliminate all the dummy switch instructions that we used before,
    //     and that should not appear in the output. The dummy switch are
    //     the first instruction of each basic block.
    for (BasicBlock &BB : *AnalyzedFunction) {

      // We exclude the unexpectedpc and anypc blocks since they have not been
      // populated with a dummy switch beforehand
      if (&BB != UnexpectedPC && &BB != AnyPC) {
        Instruction &I = *BB.begin();
        assert(isa<SwitchInst>(I) || isa<UnreachableInst>(I));
        I.eraseFromParent();
      }
    }

    // 11. We add a dummy entry basic block that is usefull for storing the
    //     alloca instructions used in each function and to avoid that the entry
    //     block has predecessors. The dummy entry basic blocks simply branches
    //     to the real entry block to have valid IR.
    BasicBlock *EntryBlock = &AnalyzedFunction->getEntryBlock();

    // If the entry block of the function has predecessors add a dummy block
    BasicBlock *Dummy = BasicBlock::Create(Context,
                                           "dummy_entry",
                                           AnalyzedFunction,
                                           &AnalyzedFunction->getEntryBlock());

    // 12. We copy the allocas at the beginning of the function where they will
    //     be used
    for (BasicBlock &BB : *AnalyzedFunction) {

      auto &InstructionList = AnalyzedFunction->getEntryBlock().getInstList();
      for (auto &OldAlloca : UsedAllocas[NewToOldBBMap[&BB]]) {
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
        InstructionList.push_back(NewAlloca);
        assert(NewAlloca->getParent() == &AnalyzedFunction->getEntryBlock());
        LocalVMap[&*OldAlloca] = NewAlloca;
      }
    }

    // Create the unconditional branch to the real entry block
    BranchInst::Create(EntryBlock, Dummy);

    // 13. Visit of the basic blocks of the function in reverse post-order and
    //     population of them with the instructions
    for (BasicBlock *NewBB : RPOT) {
      BasicBlock *OldBB = NewToOldBBMap[NewBB];

      // Actual copy of the instructions
      for (Instruction &OldInstruction : *OldBB) {
        bool IsCall = cloneInstruction(NewBB,
                                       &OldInstruction,
                                       LocalVMap);

        // If the cloneInstruction function returns true it means that we
        // emitted a function call and also the branch to the fallthrough block,
        // so we must end the inspection of the current basic block
        if (IsCall == true) {
          break;
        }
      }
    }
  }

  // 14. Create the functions and basic blocks needed for the correct execution
  //     of the exception handling mechanism

  // Populate the function_dispatcher
  populateFunctionDispatcher();

  // Retrieve the root function, we use it a lot.
  Function *Root = NewModule->getFunction("root");

  // Get the unexpectedpc block of the root function
  // TODO: do this in a more elegant way (see if we have some helper)
  BasicBlock *UnexpectedPC = nullptr;
  for (BasicBlock &BB : *Root) {
    if (GCBI.getType(&BB) == UnexpectedPCBlock) {
      UnexpectedPC = &BB;
      break;
    }
  }

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
                                                   NewModule);

  // Add the personality to the root function
  Root->setPersonalityFn(PersonalityFunction);

  // Emit at the beginning of the basic blocks identified as function entries
  // by revamb a call to the newly created corresponding LLVM function
  for (BasicBlock &BB : *Root) {
    assert(!BB.empty());

    TerminatorInst *Terminator = BB.getTerminator();
    if (MDNode *Node = Terminator->getMetadata("func.entry")) {
      StringRef FunctionNameString = getFunctionNameString(Node);
      Function *TargetFunc = NewModule->getFunction(FunctionNameString);

      // Remove the old instruction that compose the entry block (note that we
      // do not increment the iterator since the removal of the instruction
      // seems to automatically do that)
      auto It = BB.rbegin();
      while (It != BB.rend()) {
        It->eraseFromParent();
      }

      // Emit the invoke instruction
      InvokeInst::Create(TargetFunc,
                         InvokeReturnBlock,
                         CatchBB,
                         ArrayRef<Value *>(),
                         "",
                         &BB);
    }
  }

  // 15. Before emitting it in output we check that the module in passes the
  //     verifyModule pass
  raw_os_ostream Stream(dbg);
  assert(verifyModule(*NewModule, &Stream) == false);

}

bool IF::runOnFunction(Function &F) {

  // Retrieve analysis of the GeneratedCodeBasicInfo pass
  auto &GCBI = getAnalysis<GeneratedCodeBasicInfo>();

  // Clone the starting module and take note of all the mappings between
  // global objects. The new module will contain the newly generated
  // functions. We additionaly store all the mappings created in the
  // ModuleCloningVMap.
  ValueToValueMapTy ModuleCloningVMap;
  NewModule = CloneModule(F.getParent(), ModuleCloningVMap);

  // Create an object of type IsolateFunctionsImpl and run the pass
  IFI Impl(F, NewModule.get(), GCBI, ModuleCloningVMap);
  Impl.run();

  return false;
}

Module *IF::getModule() {
  // Propagate the llvm module to the meta-pass
  return NewModule.get();
}
