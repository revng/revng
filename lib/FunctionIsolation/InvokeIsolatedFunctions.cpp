/// \file InvokeIsolatedFunctions.cpp

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"

#include "revng/ABI/FunctionType/Layout.h"
#include "revng/FunctionIsolation/InvokeIsolatedFunctions.h"
#include "revng/Model/IRHelpers.h"
#include "revng/Pipeline/AllRegistries.h"
#include "revng/Pipeline/Contract.h"
#include "revng/Pipeline/Kind.h"
#include "revng/Pipeline/LLVMContainer.h"
#include "revng/Pipes/Kinds.h"
#include "revng/Pipes/RootKind.h"
#include "revng/Pipes/TaggedFunctionKind.h"

using namespace llvm;

using std::tuple;

char InvokeIsolatedFunctionsPass::ID = 0;
using Register = RegisterPass<InvokeIsolatedFunctionsPass>;
static Register
  X("invoke-isolated-functions", "Invoke Isolated Functions Pass", true, true);

struct InvokeIsolatedPipe {
  static constexpr auto Name = "invoke-isolated-functions";

  std::vector<pipeline::ContractGroup> getContract() const {
    using namespace revng::kinds;
    using namespace pipeline;

    // TODO: we're not using the following contract even if we should, probably
    //       due to some error in the contract logic.
    //       Specifically, adding this contract leads to deduce that we do *not*
    //       need /:root at beginning of the recompile-isolated step.
    ContractGroup IsolatedToRoot(Isolated,
                                 0,
                                 IsolatedRoot,
                                 0,
                                 pipeline::InputPreservation::Preserve);

    return {
      ContractGroup(Root, 0, IsolatedRoot, 0, InputPreservation::Erase)
    };
  }

  void registerPasses(llvm::legacy::PassManager &Manager) {
    Manager.add(new InvokeIsolatedFunctionsPass());
  }
};

static pipeline::RegisterLLVMPass<InvokeIsolatedPipe> Y;

class InvokeIsolatedFunctions {
private:
  using FunctionInfo = tuple<const model::Function *, BasicBlock *, Function *>;
  using FunctionMap = std::map<MetaAddress, FunctionInfo>;

private:
  const model::Binary &Binary;
  Function *RootFunction;
  Module *M;
  LLVMContext &Context;
  GeneratedCodeBasicInfo &GCBI;
  FunctionMap Map;

public:
  InvokeIsolatedFunctions(const model::Binary &Binary,
                          Function *RootFunction,
                          GeneratedCodeBasicInfo &GCBI) :
    Binary(Binary),
    RootFunction(RootFunction),
    M(RootFunction->getParent()),
    Context(M->getContext()),
    GCBI(GCBI) {

    for (const model::Function &Function : Binary.Functions()) {
      auto Name = getLLVMFunctionName(Function);
      llvm::Function *F = M->getFunction(Name);
      revng_assert(F != nullptr);
      Map[Function.Entry()] = { &Function, nullptr, F };
    }

    for (BasicBlock &BB : *RootFunction) {
      revng_assert(not BB.empty());

      MetaAddress JumpTarget = getBasicBlockJumpTarget(&BB);
      auto It = Map.find(JumpTarget);
      if (It != Map.end()) {
        get<1>(It->second) = &BB;
      }
    }
  }

  /// Create the basic blocks that are hit on exit after an invoke instruction
  BasicBlock *createInvokeReturnBlock() {
    // Create the first block
    BasicBlock *InvokeReturnBlock = BasicBlock::Create(Context,
                                                       "invoke_return",
                                                       RootFunction,
                                                       nullptr);

    BranchInst::Create(GCBI.dispatcher(), InvokeReturnBlock);

    return InvokeReturnBlock;
  }

  /// Create the basic blocks that represent the catch of the invoke instruction
  BasicBlock *createCatchBlock(BasicBlock *UnexpectedPC) {
    // Create a basic block that represents the catch part of the exception
    BasicBlock *CatchBB = BasicBlock::Create(Context,
                                             "catchblock",
                                             RootFunction,
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
                                                     M);

    // Add the personality to the root function
    RootFunction->setPersonalityFn(PersonalityFunction);

    for (auto [_, T] : Map) {
      auto [ModelF, BB, F] = T;

      // Create a new trampoline entry block and substitute it to the old entry
      // block
      BasicBlock *NewBB = BB->splitBasicBlockBefore(BB->begin());
      NewBB->getTerminator()->eraseFromParent();
      NewBB->takeName(BB);

      IRBuilder<> Builder(NewBB);

      // In case the isolated functions has arguments, provide them
      SmallVector<Value *, 4> Arguments;
      if (F->getFunctionType()->getNumParams() > 0) {
        auto ThePrototype = Binary.prototypeOrDefault(ModelF->prototype());
        auto Layout = abi::FunctionType::Layout::make(*ThePrototype);
        for (const auto &ArgumentLayout : Layout.Arguments) {
          for (model::Register::Values Register : ArgumentLayout.Registers) {
            auto Name = model::Register::getCSVName(Register);
            GlobalVariable *CSV = M->getGlobalVariable(Name, true);
            revng_assert(CSV != nullptr);
            Arguments.push_back(createLoad(Builder, CSV));
          }
        }
      }

      // Emit the invoke instruction, propagating debug info
      auto *NewInvoke = Builder.CreateInvoke(F,
                                             InvokeReturnBlock,
                                             CatchBB,
                                             Arguments);
      NewInvoke->setDebugLoc(BB->front().getDebugLoc());
    }

    // Remove all the orphan basic blocks from the root function (e.g., the
    // blocks that have been substituted by the trampoline)
    EliminateUnreachableBlocks(*RootFunction, nullptr, false);

    FunctionTags::IsolatedRoot.addTo(RootFunction);
  }
};

bool InvokeIsolatedFunctionsPass::runOnModule(Module &M) {
  using namespace pipeline;
  auto &Analysis = getAnalysis<LoadExecutionContextPass>();
  auto &RequestedTargets = Analysis.getRequestedTargets();

  if (RequestedTargets.empty())
    return false;

  revng_assert(M.getFunction("root")
               and not M.getFunction("root")->isDeclaration());

  auto &GCBI = getAnalysis<GeneratedCodeBasicInfoWrapperPass>().getGCBI();
  const auto &ModelWrapper = getAnalysis<LoadModelWrapperPass>().get();
  const model::Binary &Binary = *ModelWrapper.getReadOnlyModel();
  InvokeIsolatedFunctions TheFunction(Binary, M.getFunction("root"), GCBI);
  TheFunction.run();

  // Commit
  ExecutionContext *ExecutionContext = Analysis.get();
  ExecutionContext->commit(Target(revng::kinds::IsolatedRoot),
                           Analysis.getContainerName());

  return true;
}
