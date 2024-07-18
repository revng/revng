/// \file ArgumentUsageAnalysis.cpp

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Intrinsics.h"

#include "ArgumentUsageAnalysis.h"

static bool isVarArg(const llvm::Function &F) {
  if (not F.isVarArg())
    return false;

  // Check if the function uses llvm.va_start
  for (const llvm::Instruction &I : llvm::instructions(F))
    if (auto *Call = dyn_cast<llvm::CallBase>(&I))
      if (Call->getIntrinsicID() == llvm::Intrinsic::vastart)
        return true;

  return false;
}

namespace aua {

void ArgumentUsageAnalysis::run() {
  revng_log(Log, "Running ArgumentUsageAnalysis");
  LoggerIndent<> Indent(Log);

  // Analyze functions in post order
  llvm::CallGraph CG(M);
  for (llvm::CallGraphNode *Node : llvm::post_order(&CG)) {
    llvm::Function *F = Node->getFunction();
    if (F != nullptr and not F->isDeclaration() and not isVarArg(*F))
      analyzeFunction(*F);
  }
}

void ArgumentUsageAnalysis::analyzeFunction(llvm::Function &F) {
  revng_log(Log, "Analyzing " << F.getName());
  LoggerIndent<> Indent(Log);

  auto [It, New] = Results.insert({ &F, Function(TheContext) });
  revng_assert(New);
  Function &FunctionResults = It->second;

  // Initialize arguments
  for (auto &&[Index, Argument] : llvm::enumerate(F.args()))
    FunctionResults.set(Argument, TheContext.getArgument(Index));

  // Initialize the analysis with a conservative approach by associating each
  // instruction to FunctionOf the arguments that might affect it.
  taintAnalysis(FunctionResults, F);

  // At this point the results are correct and we could stop.

  // Do a single sweep of results refinement with more sophisticated
  // expressions
  {
    revng_log(Log, "Running analyzeInstruction");
    LoggerIndent<> Indent(Log);

    // We do one sweep in reverse-post order. This means that, before we visit
    // an instruction all of its operands will have been visited already. The
    // only exception are "recursive" phis. For those, we accept to employ
    // correct but suboptimal results compute by tainAnalysis.
    llvm::ReversePostOrderTraversal<const llvm::Function *> RPOT(&F);
    for (const llvm::BasicBlock *BB : RPOT)
      for (const llvm::Instruction &I : *BB)
        if (not I.isDebugOrPseudoInst())
          if (const Value *V = analyzeInstruction(FunctionResults, I))
            FunctionResults.set(I, *V);
  }

  {
    // Do a final pass to register memory accesses (load, store, memcpy),
    // function calls and escaped arguments
    revng_log(Log, "Running registerFunctionResults");
    LoggerIndent<> Indent(Log);
    for (llvm::BasicBlock &BB : F)
      for (llvm::Instruction &I : BB)
        if (not I.isDebugOrPseudoInst())
          registerFunctionResults(FunctionResults, I);

    // Handle functions returning void or not returning at all
    if (FunctionResults.returnValue() == nullptr)
      FunctionResults.registerReturnValue(TheContext, TheContext.getUnknown());
  }
}

void ArgumentUsageAnalysis::taintAnalysis(Function &FunctionResults,
                                          const llvm::Function &F) {
  revng_log(Log, "Running taint analysis");
  LoggerIndent<> Indent(Log);

  revng_assert(not isVarArg(F));
  revng_assert(F.arg_size() <= 64);

  // Create a map associating to each instruction a set of bits representing
  // the set of arguments that affect it
  llvm::DenseMap<const llvm::Instruction *, uint64_t> Taint;
  for (const auto &[Index, Argument] : llvm::enumerate(F.args())) {
    // Define the queue and helper lambda to enqueue more stuff
    SmallVector<const llvm::Use *> Queue;
    auto EnqueueUses = [&Queue](const llvm::Value &V) {
      for (const llvm::Use &U : V.uses()) {
        if (auto *I = dyn_cast<llvm::Instruction>(U.getUser())) {
          if (not I->isDebugOrPseudoInst() and not isa<llvm::LoadInst>(I)
              and not I->getType()->isVoidTy()) {
            Queue.push_back(&U);
          }
        }
      }
    };

    // Initialize the queue with the uses of arguments
    EnqueueUses(Argument);

    // Taint users of elements in the queue, and keep iterating until fixed
    // point. Note that we only re-enqueue only if the set of
    // arguments has grown. Given the number of arguments is finite, this will
    // converge.
    while (not Queue.empty()) {
      auto *U = Queue.pop_back_val();
      auto *I = cast<llvm::Instruction>(U->getUser());

      // If necessary, create an entry for I in the taint map
      auto It = Taint.find(I);
      if (It == Taint.end())
        It = Taint.insert({ I, 0 }).first;

      // In case of a call, if we already analyzed the callee, propagate from
      // actual argument to return value only if, according to the current
      // results, the return value is affected by the corresponding actual
      // argument.
      if (auto *Call = dyn_cast<llvm::CallInst>(I)) {
        if (const Value *Result = getCallResult(*Call)) {
          if (Call->isArgOperand(U)) {
            unsigned ActualArgumentIndex = Call->getArgOperandNo(U);
            if (not Result->collectArguments().contains(ActualArgumentIndex)) {
              continue;
            }
          }
        }
      }

      uint64_t &MapEntry = It->second;
      uint64_t OldValue = MapEntry;
      MapEntry |= 1 << Index;

      // Re-enqueue only if the entry has changed
      if (OldValue != MapEntry)
        EnqueueUses(*I);
    }
  }

  // The analysis has terminated: we now know what arguments affect each
  // instruction (modulo escaped arguments, but those are handled later)

  // Initialize all the tainted instructions with FunctionOf(Arg0, Arg3, ...)
  for (const llvm::Instruction &I : llvm::instructions(&F)) {
    if (I.getType()->isVoidTy())
      continue;

    SmallVector<const Value *, 2> Values;

    auto It = Taint.find(&I);
    if (It != Taint.end()) {
      uint64_t ArgumentSet = It->second;
      while (ArgumentSet) {
        unsigned Index = llvm::findFirstSet(ArgumentSet);
        Values.push_back(&TheContext.getArgument(Index));
        ArgumentSet &= (ArgumentSet - 1);
      }
    }

    FunctionResults.set(I, TheContext.getFunctionOf(std::move(Values)));
  }
}

const Value *
ArgumentUsageAnalysis::analyzeInstruction(Function &FunctionResults,
                                          const llvm::Instruction &I) {

  if (I.getType()->isVoidTy())
    return nullptr;

  auto &C = TheContext;
  auto Get = [&FunctionResults](const llvm::Value &V) -> const Value & {
    return FunctionResults.get(V);
  };

  switch (I.getOpcode()) {
  case llvm::Instruction::Load:
    return nullptr;

  case llvm::Instruction::Add:
    return &C.getAdd(Get(*I.getOperand(0)), Get(*I.getOperand(1)));

  case llvm::Instruction::Sub:
    return &C.getSubtract(Get(*I.getOperand(0)), Get(*I.getOperand(1)));

  case llvm::Instruction::Mul:
    return &C.getMultiply(Get(*I.getOperand(0)), Get(*I.getOperand(1)));

  case llvm::Instruction::GetElementPtr: {
    const auto *GEP = cast<llvm::GetElementPtrInst>(&I);
    const llvm::Value *Base = GEP->getPointerOperand();
    llvm::Type *BaseType = GEP->getSourceElementType();
    const Value &BaseValue = Get(*Base);

    SmallVector<llvm::Value *, 2> Indices;
    llvm::Type *NextType = BaseType;

    // Handle zero-th index
    auto ElementSize = DL.getTypeAllocSize(GEP->getSourceElementType());
    llvm::Value &FirstIndex = **GEP->idx_begin();
    const Value &IndexValue = Get(FirstIndex);
    const auto *Result = &C.getAdd(BaseValue,
                                   C.getMultiply(IndexValue,
                                                 C.getConstant(ElementSize)));

    Indices.push_back(&FirstIndex);
    NextType = llvm::GetElementPtrInst::getIndexedType(BaseType, Indices);

    // Iterate over other indices
    for (const llvm::Use &IndexUse : skip(GEP->indices(), 1)) {
      llvm::Value *Index = IndexUse.get();
      const Value &IndexValue = Get(*Index);

      if (auto *ArrayType = dyn_cast<llvm::ArrayType>(NextType)) {
        auto ElementSize = DL.getTypeAllocSize(NextType->getArrayElementType());
        auto &C = TheContext;
        Result = &C.getAdd(*Result,
                           C.getMultiply(IndexValue,
                                         C.getConstant(ElementSize)));
      } else if (auto *StructType = dyn_cast<llvm::StructType>(NextType)) {
        auto FieldIndex = getLimitedValue(Index);
        const llvm::StructLayout *Layout = DL.getStructLayout(StructType);
        auto Offset = Layout->getElementOffset(FieldIndex);
        Result = &C.getAdd(*Result, C.getConstant(Offset));
      } else {
        revng_abort();
      }

      Indices.push_back(Index);
      NextType = llvm::GetElementPtrInst::getIndexedType(BaseType, Indices);
    }

    return Result;
  }

  case llvm::Instruction::Call:
    return handleCall(FunctionResults, *cast<llvm::CallInst>(&I));

  case llvm::Instruction::PHI: {
    // Handle phis by emitting an AnyOf with one entry per incoming of the phi
    llvm::SmallVector<const Value *, 2> Alternatives;
    for (const llvm::Value *Incoming : I.operands())
      Alternatives.push_back(&Get(*Incoming));
    return &C.getAnyOf(std::move(Alternatives));
  }

  case llvm::Instruction::IntToPtr:
  case llvm::Instruction::PtrToInt:
  case llvm::Instruction::BitCast:
  case llvm::Instruction::ZExt:
  case llvm::Instruction::SExt:
  case llvm::Instruction::Trunc:
    // TODO: should we distinguish zero- vs sign-extending?
    return &Get(*I.getOperand(0));

  default: {
    // Everything else is just FunctionOf the arguments used in its operands
    llvm::SmallVector<const Value *, 2> ArgumentsInOperand;
    for (const llvm::Use &Operand : I.operands()) {
      for (const ArgumentValue *Argument :
           Get(*Operand.get()).collect<ArgumentValue>()) {
        ArgumentsInOperand.push_back(Argument);
      }
    }
    return &C.getFunctionOf(std::move(ArgumentsInOperand));
  }
  }
}

void ArgumentUsageAnalysis::registerFunctionResults(Function &FunctionResults,
                                                    llvm::Instruction &I) {
  auto Get = [&FunctionResults](const llvm::Value &V) -> const Value & {
    return FunctionResults.get(V);
  };

  switch (I.getOpcode()) {
  case llvm::Instruction::Load:
    registerLoad(FunctionResults, *cast<llvm::LoadInst>(&I));
    break;

  case llvm::Instruction::Store:
    registerStore(FunctionResults, *cast<llvm::StoreInst>(&I));
    break;

  case llvm::Instruction::Ret:
    if (cast<llvm::ReturnInst>(&I)->getNumOperands() != 0)
      FunctionResults.registerReturnValue(TheContext, Get(*I.getOperand(0)));
    break;

  case llvm::Instruction::Call: {
    auto *Call = cast<llvm::CallInst>(&I);
    if ((Call->getIntrinsicID() == llvm::Intrinsic::memcpy
         or Call->getIntrinsicID() == llvm::Intrinsic::memmove)
        and isa<llvm::ConstantInt>(Call->getArgOperand(2))) {
      registerMemcpy(FunctionResults, *Call);
    } else if ((Call->getIntrinsicID() == llvm::Intrinsic::memset)
               and isa<llvm::ConstantInt>(Call->getArgOperand(2))) {
      registerWrite(FunctionResults, Call->getArgOperandUse(0));
    } else {
      registerCall(FunctionResults, *Call);
    }
  } break;

  default:
    break;
  }
}

void ArgumentUsageAnalysis::registerCall(Function &FunctionResults,
                                         llvm::CallInst &Call) {
  auto CallSite = analyzeCallSite(Call);

  if (Log.isEnabled()) {
    Log << "registerCall: ";
    dumpCall(Log, &Call);
    Log << DoLog;
  }

  LoggerIndent<> Indent(Log);
  if (Log.isEnabled()) {
    CallSite.dump(Log);
    Log << DoLog;
  }

  if (CallSite.IsNoReturn or (CallSite.IsDeclared and CallSite.IsPure)) {
    // We do not consider noreturn calls and pure functions as escaping
    revng_log(Log, "The callee is norerutrn or pure. Ignoring call.");
  } else if (CallSite.IsIndirect or CallSite.IsDeclared
             or not CallSite.HasBeenAnalyzed or CallSite.IsVarArg) {
    revng_log(Log, "Can't handle this call.");

    for (const llvm::Value *Argument : Call.args()) {
      const Value &Value = FunctionResults.get(*Argument);
      FunctionResults.logEscapedValue("call " + getName(&Call), Value);
      FunctionResults.registerEscapedValue(Call, Value);
    }
  } else {
    revng_log(Log, "Registering.");

    // Register the call
    const Function &CalleeResults = Results.at(CallSite.Callee);
    aua::Call NewCall(Call, CalleeResults);
    for (auto &&[Index, Argument] : llvm::enumerate(Call.args())) {
      const Value &ArgumentValue = FunctionResults.get(*Argument.get());
      NewCall.registerActualArgument(Index, ArgumentValue);
    }

    FunctionResults.registerCall(std::move(NewCall));
  }
}

ArgumentUsageAnalysis::CallSite
ArgumentUsageAnalysis::analyzeCallSite(const llvm::CallInst &Call) {
  CallSite Result;

  Result.Callee = getCalledFunction(&Call);
  Result.IsIndirect = Result.Callee == nullptr;
  Result.IsDeclared = not Result.IsIndirect and Result.Callee->isDeclaration();
  Result.IsVarArg = not Result.IsIndirect and isVarArg(*Result.Callee);
  Result.HasBeenAnalyzed = Results.contains(Result.Callee);
  Result.IsNoReturn = Call.doesNotReturn()
                      or (not Result.IsIndirect
                          and Result.Callee->doesNotReturn());
  Result.IsPure = Call.doesNotAccessMemory();
  return Result;
}

const Value *ArgumentUsageAnalysis::getCallResult(const llvm::CallInst &Call) {
  auto CallSite = analyzeCallSite(Call);

  if (CallSite.IsIndirect or CallSite.IsDeclared or not CallSite.HasBeenAnalyzed
      or CallSite.IsVarArg) {
    if (Log.isEnabled()) {
      Log << "The following call cannot be handled ";
      dumpCall(Log, &Call);
      Log << "\n";
      CallSite.dump(Log);
      Log << DoLog;
    }

    // We have a call we cannot handle, return nullptr so we preserve the
    // result of the taint analysis, which is conservative since it assumes
    // the result is a FunctionOf all the arguments passed in.
    return nullptr;
  }

  return Results.at(CallSite.Callee).returnValue();
}

const Value *ArgumentUsageAnalysis::handleCall(Function &FunctionResults,
                                               const llvm::CallInst &Call) {
  // Compute result of the call
  if (const Value *Result = getCallResult(Call)) {
    llvm::DenseMap<uint64_t, const Value *> ActualArguments;
    for (auto &&[ArgumentIndex, Argument] : llvm::enumerate(Call.args()))
      ActualArguments[ArgumentIndex] = &FunctionResults.get(*Argument);
    return TheContext.replaceArguments(*Result, std::move(ActualArguments));
  }

  // This means we haven't analyzed the function yet, leave the
  // conservative value left by taintAnalysis
  return nullptr;
}

} // namespace aua
