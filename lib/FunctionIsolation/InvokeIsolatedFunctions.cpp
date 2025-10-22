/// \file InvokeIsolatedFunctions.cpp

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Linker/Linker.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Cloning.h"

#include "revng/ABI/FunctionType/Layout.h"
#include "revng/BasicAnalyses/GeneratedCodeBasicInfo.h"
#include "revng/EarlyFunctionAnalysis/ControlFlowGraphCache.h"
#include "revng/Model/IRHelpers.h"
#include "revng/Model/NameBuilder.h"
#include "revng/Pipeline/AllRegistries.h"
#include "revng/Pipeline/Contract.h"
#include "revng/Pipeline/Kind.h"
#include "revng/Pipeline/LLVMContainer.h"
#include "revng/Pipes/Kinds.h"
#include "revng/Pipes/RootKind.h"
#include "revng/Pipes/TaggedFunctionKind.h"

using namespace llvm;

class InvokeIsolatedFunctionsImpl {
private:
  using FunctionInfo = tuple<const model::Function *,
                             BasicBlock *,
                             const Function *>;
  using FunctionMap = std::map<model::Function::Key, FunctionInfo>;

private:
  const model::Binary &Binary;
  Function &RootFunction;
  Module &RootModule;
  LLVMContext &Context;
  GeneratedCodeBasicInfo GCBI;
  FunctionMap Map;

public:
  InvokeIsolatedFunctionsImpl(const model::Binary &Binary,
                              llvm::Module &RootModule,
                              const llvm::Module &FunctionModule) :
    Binary(Binary),
    RootFunction(*RootModule.getFunction("root")),
    RootModule(RootModule),
    Context(RootModule.getContext()),
    GCBI(Binary) {

    GCBI.run(RootModule);

    model::CNameBuilder NameBuilder = Binary;
    for (const model::Function &Function : Binary.Functions()) {
      auto *F = FunctionModule.getFunction(NameBuilder.llvmName(Function));
      revng_assert(F != nullptr);
      Map[Function.key()] = { &Function, nullptr, F };
    }

    for (BasicBlock &BB : RootFunction) {
      revng_assert(not BB.empty());

      MetaAddress JumpTarget = getBasicBlockJumpTarget(&BB);
      auto It = Map.find(JumpTarget);
      if (It != Map.end()) {
        revng_assert(get<1>(It->second) == nullptr);
        get<1>(It->second) = &BB;
      }
    }
  }

  /// Create the basic blocks that are hit on exit after an invoke instruction
  BasicBlock *createInvokeReturnBlock() {
    // Create the first block
    BasicBlock *InvokeReturnBlock = BasicBlock::Create(Context,
                                                       "invoke_return",
                                                       &RootFunction,
                                                       nullptr);

    BranchInst::Create(GCBI.dispatcher(), InvokeReturnBlock);

    return InvokeReturnBlock;
  }

  /// Create the basic blocks that represent the catch of the invoke instruction
  BasicBlock *createCatchBlock(BasicBlock *UnexpectedPC) {
    // Create a basic block that represents the catch part of the exception
    BasicBlock *CatchBB = BasicBlock::Create(Context,
                                             "catchblock",
                                             &RootFunction,
                                             nullptr);

    // Create a builder object (this pipeline branch never leads to decompiled
    // code, so who cares about the debug information).
    revng::NonDebugInfoCheckingIRBuilder Builder(Context);
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
    auto *NullPtr = ConstantPointerNull::get(Type::getInt8PtrTy(Context));
    LandingPad->addClause(NullPtr);

    Builder.CreateBr(UnexpectedPC);

    return CatchBB;
  }

  void run() {
    // Get the unexpectedpc block of the root function
    BasicBlock *UnexpectedPC = GCBI.unexpectedPC();

    // Instantiate the basic block structure that handles the control flow after
    // an invoke
    BasicBlock *InvokeReturnBlock = createInvokeReturnBlock();

    // Instantiate the basic block structure that represents the catch of the
    // invoke, please remember that this is not used at the moment (exceptions
    // are handled in a customary way from the standard exit control flow path)
    BasicBlock *CatchBB = createCatchBlock(UnexpectedPC);

    // Declaration of an ad-hoc personality function that is implemented in the
    // support.c source file
    auto *PersonalityFT = FunctionType::get(Type::getInt32Ty(Context), true);

    Function *PersonalityFunction = Function::Create(PersonalityFT,
                                                     Function::ExternalLinkage,
                                                     "__gxx_personality_v0",
                                                     RootModule);

    // Add the personality to the root function
    RootFunction.setPersonalityFn(PersonalityFunction);

    for (auto &&[_, T] : Map) {
      auto &&[ModelF, BB, F] = T;

      // Create a new trampoline entry block and substitute it to the old entry
      // block
      BasicBlock *NewBB = BB->splitBasicBlockBefore(BB->begin());
      NewBB->getTerminator()->eraseFromParent();
      NewBB->takeName(BB);

      // This pipeline branch never leads to decompiled code, so who cares about
      // the debug information.
      revng::NonDebugInfoCheckingIRBuilder Builder(NewBB);

      // In case the isolated functions has arguments, provide them
      SmallVector<Value *, 4> Arguments;
      if (F->getFunctionType()->getNumParams() > 0) {
        auto ThePrototype = Binary.prototypeOrDefault(ModelF->prototype());
        auto Layout = abi::FunctionType::Layout::make(*ThePrototype);
        for (const auto &ArgumentLayout : Layout.Arguments) {
          for (model::Register::Values Register : ArgumentLayout.Registers) {
            auto Name = model::Register::getCSVName(Register);
            GlobalVariable *CSV = RootModule.getGlobalVariable(Name, true);
            revng_assert(CSV != nullptr);
            Arguments.push_back(createLoad(Builder, CSV));
          }
        }
      }

      llvm::Function *
        NewDeclaration = llvm::Function::Create(F->getFunctionType(),
                                                llvm::Function::ExternalLinkage,
                                                F->getName(),
                                                RootModule);

      // Emit the invoke instruction, propagating debug info
      auto *NewInvoke = Builder.CreateInvoke(NewDeclaration,
                                             InvokeReturnBlock,
                                             CatchBB,
                                             Arguments);
      NewInvoke->setDebugLoc(BB->front().getDebugLoc());
    }

    // Remove all the orphan basic blocks from the root function (e.g., the
    // blocks that have been substituted by the trampoline)
    EliminateUnreachableBlocks(RootFunction, nullptr, false);

    FunctionTags::IsolatedRoot.addTo(&RootFunction);
  }
};

static void populateFunctionDispatcher(const model::Binary &Binary,
                                       llvm::Module &Module) {
  GeneratedCodeBasicInfo GCBI(Binary);
  GCBI.run(Module);

  llvm::LLVMContext &Context = Module.getContext();
  llvm::Function *FunctionDispatcher = getIRHelper("function_dispatcher",
                                                   Module);
  BasicBlock *Dispatcher = BasicBlock::Create(Context,
                                              "function_dispatcher",
                                              FunctionDispatcher,
                                              nullptr);

  BasicBlock *Unexpected = BasicBlock::Create(Context,
                                              "unexpectedpc",
                                              FunctionDispatcher,
                                              nullptr);
  revng::NonDebugInfoCheckingIRBuilder UnreachableBuilder(Unexpected);
  UnreachableBuilder.CreateUnreachable();
  setBlockType(Unexpected->getTerminator(), BlockType::UnexpectedPCBlock);

  // TODO: the checks should be enabled conditionally based on the user.
  revng::NonDebugInfoCheckingIRBuilder Builder(Context);

  // Create all the entries of the dispatcher
  ProgramCounterHandler::DispatcherTargets Targets;
  for (llvm::Function &F : Module.functions()) {
    if (not FunctionTags::Isolated.isTagOf(&F))
      continue;

    MetaAddress Address = getMetaAddressOfIsolatedFunction(F);
    BasicBlock *Trampoline = BasicBlock::Create(Context,
                                                F.getName() + "_trampoline",
                                                FunctionDispatcher,
                                                nullptr);
    Targets.emplace_back(Address, Trampoline);

    Builder.SetInsertPoint(Trampoline);
    Builder.CreateCall(&F);
    Builder.CreateRetVoid();
  }

  // Create switch
  Builder.SetInsertPoint(Dispatcher);
  GCBI.programCounterHandler()->buildDispatcher(Targets,
                                                Builder,
                                                Unexpected,
                                                {});
}

struct InvokeIsolatedPipe {
  static constexpr auto Name = "invoke-isolated-functions";

  std::vector<pipeline::ContractGroup> getContract() const {
    using namespace revng;
    using namespace pipeline;
    return { ContractGroup({ Contract(kinds::Root,
                                      0,
                                      kinds::IsolatedRoot,
                                      2,
                                      InputPreservation::Preserve),
                             Contract(kinds::Isolated,
                                      1,
                                      kinds::IsolatedRoot,
                                      2,
                                      InputPreservation::Preserve) }) };
  }

public:
  void run(pipeline::ExecutionContext &EC,
           pipeline::LLVMContainer &InputRootContainer,
           pipeline::LLVMContainer &FunctionContainer,
           pipeline::LLVMContainer &OutputRootContainer) {
    // Clone the container
    OutputRootContainer.cloneFrom(InputRootContainer);

    populateFunctionDispatcher(*revng::getModelFromContext(EC),
                               FunctionContainer.getModule());
    InvokeIsolatedFunctionsImpl Impl(*revng::getModelFromContext(EC),
                                     OutputRootContainer.getModule(),
                                     FunctionContainer.getModule());
    Impl.run();

    const llvm::Module &FunctionModule = FunctionContainer.getModule();
    linkModules(llvm::CloneModule(FunctionModule),
                OutputRootContainer.getModule());

    EC.commit(pipeline::Target(revng::kinds::IsolatedRoot),
              OutputRootContainer.name());
  }
};

static pipeline::RegisterPipe<InvokeIsolatedPipe> Y;
