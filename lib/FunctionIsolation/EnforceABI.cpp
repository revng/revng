/// Promotes global variables CSV to function arguments or local variables,
/// according to the ABI analysis.

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/raw_os_ostream.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"

#include "revng/ABI/DefaultFunctionPrototype.h"
#include "revng/ABI/FunctionType/Layout.h"
#include "revng/ADT/LazySmallBitVector.h"
#include "revng/ADT/SmallMap.h"
#include "revng/BasicAnalyses/GeneratedCodeBasicInfo.h"
#include "revng/EarlyFunctionAnalysis/CallEdge.h"
#include "revng/EarlyFunctionAnalysis/ControlFlowGraphCache.h"
#include "revng/FunctionIsolation/EnforceABI.h"
#include "revng/FunctionIsolation/IsolateFunctions.h"
#include "revng/FunctionIsolation/StructInitializers.h"
#include "revng/Model/FunctionTags.h"
#include "revng/Model/IRHelpers.h"
#include "revng/Model/Register.h"
#include "revng/Model/TypeDefinition.h"
#include "revng/Support/BlockType.h"
#include "revng/Support/IRBuilder.h"
#include "revng/Support/IRHelpers.h"
#include "revng/Support/MetaAddress.h"
#include "revng/Support/OpaqueFunctionsPool.h"

using namespace llvm;

class EnforceABI {
private:
  using UsedRegisters = abi::FunctionType::UsedRegisters;
  using CFG = efa::ControlFlowGraph;

private:
  const model::Binary &Binary;
  llvm::Module &M;
  const model::Function &ModelFunction;
  llvm::Function &OldFunction;
  const revng::pypeline::CFGMap &CFGMap;
  GeneratedCodeBasicInfo GCBI;

  std::map<Function *, Function *> OldToNew;
  std::vector<Function *> OldFunctions;
  Function *FunctionDispatcher = nullptr;
  StructInitializers Initializers;

public:
  EnforceABI(const model::Binary &Binary,
             const model::Function &ModelFunction,
             llvm::Function &OldFunction,
             const revng::pypeline::CFGMap &CFGMap) :
    Binary(Binary),
    M(*OldFunction.getParent()),
    ModelFunction(ModelFunction),
    OldFunction(OldFunction),
    CFGMap(CFGMap),
    GCBI(Binary, M),
    Initializers(&M) {}

  ~EnforceABI() = default;

public:
  void run();

private:
  Function *getOrCreateNewFunction(Function &OldFunction,
                                   const UsedRegisters &UsedRegisters);

  Function *recreateFunction(Function &OldFunction,
                             const UsedRegisters &UsedRegisters);

  void createPrologue(Function *NewFunction,
                      const UsedRegisters &UsedRegisters);

  void handleRegularFunctionCall(CallInst *Call);
  CallInst *generateCall(revng::IRBuilder &Builder,
                         MetaAddress Entry,
                         FunctionCallee Callee,
                         const efa::BasicBlock &CallSiteBlock,
                         const efa::CallEdge &CallSite);
};

static Logger EnforceABILog("enforce-abi");

void EnforceABI::run() {
  FunctionDispatcher = functionOrNull(FunctionDispatcherHelper.get(M));

  if (FunctionDispatcher != nullptr)
    OldFunctions.push_back(FunctionDispatcher);

  // TODO: make this lazy and push to runOnFunction

  // Recreate dynamic functions with arguments
  for (const model::DynamicFunction &FunctionModel :
       Binary.ImportedDynamicFunctions()) {
    // TODO: have an API to go from model to llvm::Function
    auto OldFunctionName = (Twine("dynamic_") + FunctionModel.Name()).str();
    Function *OldFunction = M.getFunction(OldFunctionName);
    if (not OldFunction)
      continue;
    OldFunctions.push_back(OldFunction);

    const auto *ProtoT = Binary.prototypeOrDefault(FunctionModel.prototype());
    revng_assert(ProtoT != nullptr);
    auto UsedRegisters = abi::FunctionType::usedRegisters(*ProtoT);
    Function *NewFunction = recreateFunction(*OldFunction, UsedRegisters);

    // EnforceABI currently does not support execution
    NewFunction->deleteBody();

    // Copy metadata
    NewFunction->copyMetadata(OldFunction, 0);

    OldToNew[OldFunction] = NewFunction;
  }

  // Recreate the function with the right prototype and the function prologue
  const auto *ProtoT = Binary.prototypeOrDefault(ModelFunction.prototype());
  revng_assert(ProtoT != nullptr);
  auto UsedRegisters = abi::FunctionType::usedRegisters(*ProtoT);
  Function *NewFunction = getOrCreateNewFunction(OldFunction, UsedRegisters);

  // Collect function calls
  SmallVector<CallInst *, 16> Calls;
  for (Instruction &I : llvm::instructions(NewFunction)) {
    if (auto *Call = dyn_cast<CallInst>(&I)) {
      // All the calls are still calling the old function
      Function *Callee = getCalledFunction(Call);

      // Mainly for dynamic functions
      bool Recreated = OldToNew.contains(Callee);
      if (Recreated or isCallToIsolatedFunction(Call)
          or Callee == FunctionDispatcher) {
        Calls.emplace_back(Call);
      }
    }
  }

  // Actually process function calls
  for (CallInst *Call : Calls)
    handleRegularFunctionCall(Call);

  // Drop all the old functions, after we stole all of its blocks
  for (Function *OldFunction : OldFunctions)
    eraseFromParent(OldFunction);

  // Quick and dirty DCE
  for (auto &&[_, F] : OldToNew)
    if (not F->isDeclaration())
      EliminateUnreachableBlocks(*F, nullptr, false);

  revng::verify(&M);
}

using UsedRegisters = abi::FunctionType::UsedRegisters;
static std::pair<Type *, SmallVector<Type *>>
getLLVMReturnTypeAndArguments(llvm::Module *M, const UsedRegisters &Registers) {
  LLVMContext &Context = M->getContext();
  auto ArgumentsTypes = toLLVMTypes(Context, Registers.Arguments);
  auto ReturnTypes = toLLVMTypes(Context, Registers.ReturnValues);

  // Create the return type
  Type *ReturnType = Type::getVoidTy(Context);
  if (ReturnTypes.size() == 0)
    ReturnType = Type::getVoidTy(Context);
  else if (ReturnTypes.size() == 1)
    ReturnType = ReturnTypes[0];
  else
    ReturnType = StructType::get(Context, ReturnTypes, true);

  // Create new function
  return { ReturnType, ArgumentsTypes };
}

Function *
EnforceABI::getOrCreateNewFunction(Function &OldFunction,
                                   const UsedRegisters &UsedRegisters) {
  // NOTE: all the model *must* be read above this line!
  //       If we don't do this, we will break invalidation tracking information.
  auto It = OldToNew.find(&OldFunction);
  if (It != OldToNew.end())
    return It->second;

  OldFunctions.push_back(&OldFunction);

  Function *NewFunction = recreateFunction(OldFunction, UsedRegisters);
  FunctionTags::ABIEnforced.addTo(NewFunction);

  if (not NewFunction->isDeclaration())
    createPrologue(NewFunction, UsedRegisters);

  OldToNew[&OldFunction] = NewFunction;

  return NewFunction;
}

Function *EnforceABI::recreateFunction(Function &OldFunction,
                                       const abi::FunctionType::UsedRegisters
                                         &Registers) {
  // Create new function
  auto &&[NewReturnType,
          NewArguments] = getLLVMReturnTypeAndArguments(&M, Registers);

  auto *Result = changeFunctionType(OldFunction, NewReturnType, NewArguments);
  revng_assert(Result->arg_size() == Registers.Arguments.size());

  for (size_t Index = 0; model::Register::Values Register : Registers.Arguments)
    Result->getArg(Index++)->setName(model::Register::getName(Register));

  return Result;
}

static GlobalVariable *tryGetCSV(Module *M, model::Register::Values Register) {
  auto Name = model::Register::getCSVName(Register);
  return M->getGlobalVariable(Name, true);
}

static Value *loadCSVOrUndef(revng::IRBuilder &Builder,
                             Module *M,
                             model::Register::Values Register) {
  GlobalVariable *CSV = tryGetCSV(M, Register);
  if (CSV == nullptr) {
    auto Size = model::Register::getSize(Register);
    auto *Type = IntegerType::get(M->getContext(), Size * 8);
    return UndefValue::get(Type);
  } else {
    return Builder.createLoad(CSV);
  }
}

static std::pair<Type *, Constant *>
getCSVOrUndef(Module *M, model::Register::Values Register) {
  GlobalVariable *CSV = tryGetCSV(M, Register);
  if (CSV == nullptr) {
    auto Size = model::Register::getSize(Register);
    auto *Type = IntegerType::get(M->getContext(), Size * 8);
    return { Type, UndefValue::get(PointerType::get(M->getContext(), 0)) };
  } else {
    return { CSV->getValueType(), CSV };
  }
}

void EnforceABI::createPrologue(Function *NewFunction,
                                const abi::FunctionType::UsedRegisters
                                  &UsedRegisters) {
  SmallVector<Constant *, 8> ArgumentCSVs;
  SmallVector<std::pair<Type *, Constant *>, 8> ReturnCSVs;

  // We sort arguments by their CSV name
  auto &&[ArgumentRegisters, ReturnValueRegisters] = UsedRegisters;
  for (model::Register::Values Register : ArgumentRegisters)
    ArgumentCSVs.push_back(getCSVOrUndef(&M, Register).second);
  for (model::Register::Values Register : ReturnValueRegisters)
    ReturnCSVs.push_back(getCSVOrUndef(&M, Register));

  // Store arguments to CSVs
  BasicBlock &Entry = NewFunction->getEntryBlock();
  revng::IRBuilder StoreBuilder(Entry.getTerminator());
  for (const auto &[TheArgument, CSV] : zip(NewFunction->args(), ArgumentCSVs))
    StoreBuilder.CreateStore(&TheArgument, CSV);

  // Build the return value
  if (ReturnCSVs.size() != 0) {
    for (BasicBlock &BB : *NewFunction) {
      if (auto *Return = dyn_cast<ReturnInst>(BB.getTerminator())) {
        revng::IRBuilder Builder(Return);
        std::vector<Value *> ReturnValues;
        for (auto &&[Type, ReturnCSV] : ReturnCSVs)
          ReturnValues.push_back(Builder.CreateLoad(Type, ReturnCSV));

        if (ReturnValues.size() == 1)
          Builder.CreateRet(ReturnValues[0]);
        else
          Initializers.createReturn(Builder, ReturnValues);

        eraseFromParent(Return);
      }
    }
  }
}

void EnforceABI::handleRegularFunctionCall(CallInst *Call) {
  revng_assert(Call->getDebugLoc());

  // Identify the corresponding call site in the model
  Function *CallerFunction = Call->getParent()->getParent();
  MetaAddress Entry = getMetaAddressOfIsolatedFunction(*CallerFunction);
  const efa::FunctionBundle &Bundle = *CFGMap.getElement(ObjectID(Entry));

  // The call site can come from a function inlined into this one, in which
  // case it is described by that function, not by this one
  auto &&[FM, CallerBlock] = Bundle.findBlock(Call);
  revng_assert(CallerBlock != nullptr);

  // Find the CallEdge
  const efa::CallEdge *CallSite = nullptr;
  for (const auto &Edge : CallerBlock->Successors()) {
    using namespace efa::FunctionEdgeType;
    CallSite = dyn_cast<efa::CallEdge>(Edge.get());
    if (CallSite != nullptr)
      break;
  }
  revng_assert(CallSite != nullptr);

  // Get or create the called function
  Function *Callee = cast<Function>(skipCasts(Call->getCalledOperand()));
  bool IsDirect = (Callee != FunctionDispatcher);
  bool IsDynamic = not CallSite->DynamicFunction().empty();
  if (IsDynamic) {
    Callee = OldToNew.at(Callee);
  } else if (IsDirect) {
    MetaAddress CalleeAddress = CallSite->Destination().notInlinedAddress();
    const model::Function &ModelFunc = Binary.Functions().at(CalleeAddress);
    const auto *Prototype = Binary.prototypeOrDefault(ModelFunc.prototype());
    revng_assert(Prototype != nullptr);
    auto UsedRegisters = abi::FunctionType::usedRegisters(*Prototype);

    Callee = getOrCreateNewFunction(*Callee, UsedRegisters);
  }

  // Note that currently, in case of indirect call, we emit a call to a
  // placeholder function that will throw an exception. If exceptions are
  // correctly supported post enforce-abi, and the ABI data is correct, this
  // should work. However this is not very efficient.
  //
  // Alternatives:
  //
  // 1. Emit an inline dispatcher that calls all the compatible function (i.e.,
  //    they take a subset of the call site's arguments and return a superset
  //    of the call site's return values).
  // 2. We have a dedicated outlined dispatcher that takes all the arguments of
  //    the call site, plus all the registers of the return values. Under the
  //    assumption that each return value of the call site is either a return
  //    value of the callee or is preserved by the callee, we can fill each
  //    return value using the callee's return value or the argument
  //    representing the value of that register before the call.
  //    In case the call site expects a return value that is neither a return
  //    value nor a preserved register or the callee, we exclude it from the
  ///   switch.

  // Generate the call
  revng::IRBuilder Builder(Call);
  CallInst *NewCall = generateCall(Builder,
                                   FM->Entry(),
                                   Callee,
                                   *CallerBlock,
                                   *CallSite);
  NewCall->copyMetadata(*Call);
  NewCall->setAttributes(Call->getAttributes());

  // Set PC to the expected value
  GCBI.programCounterHandler()->setPC(Builder, CallerBlock->ID().start());

  // Drop the original call
  eraseFromParent(Call);
}

static FunctionCallee
toFunctionPointer(revng::IRBuilder &B, Value *V, FunctionType *FT) {
  Module *M = getModule(B.GetInsertBlock());
  const auto &DL = M->getDataLayout();
  IntegerType *IntPtrTy = DL.getIntPtrType(M->getContext());
  Value *Callee = B.CreateIntToPtr(V, FT->getPointerTo());
  return FunctionCallee(FT, Callee);
}

CallInst *EnforceABI::generateCall(revng::IRBuilder &Builder,
                                   MetaAddress Entry,
                                   FunctionCallee Callee,
                                   const efa::BasicBlock &CallSiteBlock,
                                   const efa::CallEdge &CallSite) {
  using model::NamedTypedRegister;
  using model::RawFunctionDefinition;

  revng_assert(Callee.getCallee() != nullptr);

  llvm::SmallVector<Value *, 8> Arguments;
  llvm::SmallVector<Constant *, 8> ReturnCSVs;

  auto *Prototype = getPrototype(Binary, Entry, CallSiteBlock.ID(), CallSite);
  revng_assert(Prototype != nullptr);
  auto Registers = abi::FunctionType::usedRegisters(*Prototype);

  bool IsIndirect = (Callee.getCallee() == FunctionDispatcher);
  if (IsIndirect) {
    // Create a new `indirect_placeholder` function with the specific function
    // type we need
    Value *PC = GCBI.programCounterHandler()->loadJumpablePC(Builder);
    auto &&[ReturnType, Arguments] = getLLVMReturnTypeAndArguments(&M,
                                                                   Registers);
    auto *NewType = FunctionType::get(ReturnType, Arguments, false);
    Callee = toFunctionPointer(Builder, PC, NewType);
  } else {
    BasicBlock *InsertBlock = Builder.GetInsertPoint()->getParent();
    revng_log(EnforceABILog,
              "Emitting call to " << getName(Callee.getCallee()) << " from "
                                  << getName(InsertBlock));
  }

  //
  // Collect arguments and returns
  //
  for (model::Register::Values Register : Registers.Arguments)
    Arguments.push_back(loadCSVOrUndef(Builder, &M, Register));

  for (model::Register::Values Register : Registers.ReturnValues)
    ReturnCSVs.push_back(getCSVOrUndef(&M, Register).second);

  //
  // Produce the call
  //
  auto *Result = Builder.CreateCall(Callee, Arguments);
  revng_assert(Result->getDebugLoc());
  setStringMetadata(Result,
                    PrototypeMDName,
                    Binary.getTypeDefinitionReference(Prototype->key())
                      .toString());

  if (ReturnCSVs.size() != 1) {
    unsigned I = 0;
    for (Constant *ReturnCSV : ReturnCSVs) {
      Builder.CreateStore(Builder.CreateExtractValue(Result, { I }), ReturnCSV);
      I++;
    }
  } else {
    Builder.CreateStore(Result, ReturnCSVs[0]);
  }

  return Result;
}

namespace revng::pypeline::piperuns {

void EnforceABI::runOnLLVMFunction(const model::Function &Function,
                                   llvm::Function &LLVMFunction) {
  ::EnforceABI Impl(Binary, Function, LLVMFunction, CFG);
  Impl.run();
}

} // namespace revng::pypeline::piperuns
