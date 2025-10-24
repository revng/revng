/// \file CPUStateUsage.cpp

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <cstdint>
#include <iterator>
#include <optional>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/ModuleSlotTracker.h"

#include "revng/ADT/RecursiveCoroutine.h"
#include "revng/HelperArgumentsAnalysis/Annotation.h"
#include "revng/Support/IRHelpers.h"

#include "CPUStateUsage.h"
#include "Function.h"

namespace aua {

static RecursiveCoroutine<PointerSet> fromValueImpl(const Value &V) {
  switch (V.kind()) {
  case Value::Kind::Invalid:
    revng_abort();
  case Value::Kind::Constant:
    rc_return PointerSet::fromConstant(llvm::cast<ConstantValue>(V).value());

  case Value::Kind::AnyOf: {
    PointerSet Result;
    for (const Value *Operand : llvm::cast<AnyOfValue>(V).operands())
      Result.merge(rc_recur fromValueImpl(*Operand));
    rc_return Result;
  }

  case Value::Kind::BinaryOperator: {
    auto &Operator = llvm::cast<BinaryOperatorValue>(V);
    PointerSet LHS = rc_recur fromValueImpl(Operator.firstOperand());
    PointerSet RHS = rc_recur fromValueImpl(Operator.secondOperand());

    switch (Operator.type()) {
    case BinaryOperatorValue::Invalid:
      revng_abort();
    case BinaryOperatorValue::Add:
      rc_return LHS.add(RHS);
    case BinaryOperatorValue::Subtract:
      rc_return LHS.add(RHS.negate());
    case BinaryOperatorValue::Multiply:
      // TODO: we could restrict the possible values using SCEV/LVI
      if (const int64_t *Value = LHS.getConstant())
        rc_return PointerSet::fromStrided(*Value);
      else if (const int64_t *Value = RHS.getConstant())
        rc_return PointerSet::fromStrided(*Value);
      else
        rc_return PointerSet::unknown();
    }
  }

  case Value::Kind::Argument:
    rc_return PointerSet::unknown();
  case Value::Kind::FunctionOf:
    rc_return PointerSet::fromStrided(1);
  }
}

PointerSet PointerSet::fromValue(const Value &V) {
  return fromValueImpl(V);
}

[[nodiscard]] PointerSet PointerSet::add(const PointerSet &Other) const {
  PointerSet Result;

  for (int64_t LHS : Offsets)
    for (int64_t RHS : Other.Offsets)
      Result.Offsets.insert(LHS + RHS);

  std::set_union(Strides.begin(),
                 Strides.end(),
                 Other.Strides.begin(),
                 Other.Strides.end(),
                 std::inserter(Result.Strides, Result.Strides.end()));

  return Result;
}

[[nodiscard]] std::string PointerSet::toString() const {
  std::string Result = "{";
  for (int64_t Offset : Offsets)
    Result += " " + std::to_string(Offset);
  Result += " }";

  for (auto &&[Index, Stride] : llvm::enumerate(Strides))
    Result += " + i" + std::to_string(Index) + " * " + std::to_string(Stride);

  return Result;
}

std::optional<llvm::DenseSet<uint64_t>>
CPUStateUsageAnalysis::computeAccessesInRoot(const Value &Offset) const {
  llvm::DenseSet<uint64_t> Result;

  revng_log(Log, "PointerSet: " << PointerSet::fromValue(Offset).toString());

  auto Pointers = PointerSet::fromValue(Offset).enumerate();

  revng_assert(Pointers.size() > 0);

  for (const PointerSet &Pointer : Pointers) {
    revng_log(Log, "Considering " << Pointer.toString());
    LoggerIndent<> Indent(Log);

    // Collect the number of elements in arrays whose elements match the
    // strides in Pointer
    using namespace llvm;
    APInt Offset(64, Pointer.offset());
    llvm::DenseMap<int64_t, uint64_t> StrideToArraySize;

    Type *CurrentType = &RootType;
    std::optional<APInt> MaybeIndex = APInt(32, 0);
    while (MaybeIndex.has_value()) {
      if (Log.isEnabled()) {
        Log << "Considering offset " << Offset.getLimitedValue() << " in type ";
        CurrentType->print(*Log.getAsLLVMStream(), true);
        Log << DoLog;
      }

      if (auto *Array = dyn_cast<ArrayType>(CurrentType)) {
        auto ElementSize = DL.getTypeAllocSize(Array->getElementType());
        revng_assert(StrideToArraySize.count(ElementSize) == 0);
        auto Elements = Array->getArrayNumElements();
        revng_log(Log,
                  "Registering array of " << Elements << " elements of size "
                                          << ElementSize);
        StrideToArraySize[ElementSize] = Elements;
      } else if (auto *Struct = dyn_cast<StructType>(CurrentType)) {
        auto Elements = Struct->getNumElements();
        if (Elements > 1) {
          llvm::Type *FirstType = Struct->getElementType(0);
          auto IsAsFirst = [FirstType](llvm::Type *ElementType) {
            return ElementType == FirstType;
          };
          if (llvm::all_of(Struct->elements(), IsAsFirst)) {
            auto ElementSize = DL.getTypeAllocSize(FirstType);
            revng_assert(StrideToArraySize.count(ElementSize) == 0);
            revng_log(Log,
                      "Registering struct of "
                        << Elements << " elements of size " << ElementSize);
            StrideToArraySize[ElementSize] = Elements;
          }
        }
      }

      MaybeIndex = DL.getGEPIndexForOffset(CurrentType, Offset);
      if (MaybeIndex) {
        revng_log(Log, "Found at index " << MaybeIndex->getLimitedValue());
      } else {
        revng_log(Log, "Not found");
        break;
      }
    }

    // Enumerate all the possible offsets considering all the arrays
    struct ArrayEntry {
      uint64_t Index = 0;
      const uint64_t ArraySize = 0;
      const int64_t Stride = 0;
    };
    SmallVector<ArrayEntry, 2> WorkList;

    // Check if we found all the strides
    if (Pointer.strides().size() > 0) {

      // Identify the strides we couldn't assign
      SmallVector<int64_t, 2> MissingStrides;
      for (int64_t Stride : Pointer.strides()) {
        auto It = StrideToArraySize.find(Stride);
        if (It == StrideToArraySize.end()) {
          revng_log(Log,
                    "Couldn't find an array with elements of size "
                      << Stride << ". Will try with a compatible array later.");
          MissingStrides.push_back(Stride);
          continue;
        }
        StrideToArraySize.erase(It);
        WorkList.push_back({ 0, It->second, Stride });
      }

      // Try to assign to a compatible array, if it had a different array
      // element size
      for (int64_t Stride : MissingStrides) {
        bool Found = false;
        for (auto &&[ElementSize, Elements] : StrideToArraySize) {
          auto ArraySize = Elements * ElementSize;
          if (ArraySize % Stride == 0) {
            WorkList.push_back({ 0, ArraySize / Stride, Stride });
            StrideToArraySize.erase(ElementSize);
            revng_log(Log,
                      "Stride " << Stride << " assigned to an array of "
                                << Elements << " of size " << ElementSize);
            Found = true;
            break;
          }
        }

        if (not Found) {
          revng_log(Log,
                    "Couldn't find an array with elements compatible with "
                    "stride "
                      << Stride);
          return std::nullopt;
        }
      }

    } else {
      WorkList.push_back({ 0, 1, 0 });
    }

    bool Done = WorkList.empty();
    while (not Done) {
      // Compute new entry
      int64_t NewOffset = Pointer.offset();
      for (ArrayEntry &Entry : WorkList)
        NewOffset += Entry.Index * Entry.Stride;
      Result.insert(NewOffset);

      // Move forward
      Done = true;
      for (ArrayEntry &Entry : WorkList) {
        ++Entry.Index;
        if (Entry.Index == Entry.ArraySize) {
          Entry.Index = 0;
        } else {
          Done = false;
          break;
        }
      }
    }
  }

  return Result;
}

void CPUStateUsageAnalysis::analyze(llvm::Function &Function) {
  llvm::Task T(2, "Analyze CPU state usage of " + Function.getName());
  FastValuePrinter Printer(*Function.getParent());
  revng_log(Log, "Collecting interprocedural data in " << Function.getName());
  LoggerIndent<> Indent(Log);

  revng_log(Log, "Collecting global results");
  T.advance("Collecting global results");
  CPUStateUsage HelperResult;
  HelperResult.RawAUAResults = collectGlobalAUAResults(Function);

  T.advance("Processing arguments");
  revng_log(Log, "Processing arguments of " << Function.getName());
  LoggerIndent<> Indent2(Log);

  for (auto &&[ArgumentIndex, Argument] : llvm::enumerate(Function.args())) {
    auto &Offsets = Initializer.getOffsetsFor(Argument);
    SmallVector<const aua::Value *, 2> OffsetsValues;
    for (uint64_t Offset : Offsets)
      OffsetsValues.push_back(&TheContext.getConstant(Offset));

    if (Offsets.size() == 0)
      continue;

    if (Log.isEnabled()) {
      Log << "Processing argument " << ArgumentIndex << " of "
          << Function.getName() << " which can be at the following offsets:";
      for (uint64_t Offset : Offsets) {
        Log << " " << Offset;
      }
      Log << DoLog;
    }

    LoggerIndent<> Indent(Log);

    // If the argument is escaping, bail out
    bool Escapes = false;
    for (const EscapedArgument &Escaper :
         HelperResult.RawAUAResults.EscapedArguments) {
      // TODO: we could use lower_bound with
      //       EscapedArgument(nullptr, Escaper.index());
      if (Escaper.index() == ArgumentIndex) {
        HelperCPUStateUsage[&Function] = CPUStateUsage::escapes();
        llvm::Instruction *I = &Escaper.location();
        Escaping.insert(I);
        Escapes = true;
        revng_log(Log,
                  "Argument escapes at " << Printer.toString(*I) << " in "
                                         << Function.getName());
      }
    }

    if (Escapes)
      return;

    revng_log(Log,
              "Consindering " << HelperResult.RawAUAResults.Accesses.size()
                              << " memory accesses");
    for (auto &[Access, Count] : HelperResult.RawAUAResults.Accesses) {

      if (not Access.start().collectArguments().contains(ArgumentIndex)) {
        revng_log(Log,
                  "Ignoring memory access since it's not based on argument "
                    << ArgumentIndex);
        continue;
      }

      revng_log(Log,
                "Considering access starting at "
                  << Access.start().toString() << " in "
                  << getName(Access.location()));

      llvm::DenseMap<uint64_t, const aua::Value *> Replacements;

      auto &C = TheContext;
      SmallVector<const aua::Value *, 2> Copy = OffsetsValues;
      Replacements[ArgumentIndex] = &C.getAnyOf(std::move(Copy));
      const Value &Pointer = *C.replaceArguments(Access.start(), Replacements);
      revng_log(Log, "Folding it to " << Pointer.toString());
      auto MaybeOffsets = computeAccessesInRoot(Pointer);

      if (not MaybeOffsets) {
        auto *I = cast<llvm::Instruction>(Access.location().getUser());
        revng_log(Log,
                  "Marking argument as escaped: we couldn't compute the set "
                  "of offsets for "
                    << getName(I) << " in "
                    << I->getParent()->getParent()->getName());
        HelperCPUStateUsage[&Function] = CPUStateUsage::escapes();
        Escaping.insert(I);
        return;
      }

      revng_assert(MaybeOffsets->size() > 0);

      // Update the number of accesses this expands to
      revng_assert(Count == 0);
      Count = MaybeOffsets->size();

      if (Log.isEnabled()) {
        Log << "Offsets: {";
        for (uint64_t Offset : *MaybeOffsets) {
          Log << " " << Offset;
        }
        Log << " }" << DoLog;
      }

      auto Size = size(Access);
      bool IsWrite = Access.isWrite();

      for (uint64_t Offset : *MaybeOffsets)
        MemoryAccessOffsets[&Access.location()].insert({ Offset, Size });

      if (IsWrite) {
        for (uint64_t Offset : *MaybeOffsets) {
          HelperResult.Writes.insert({ Offset, Size });
        }
      } else {
        for (uint64_t Offset : *MaybeOffsets) {
          HelperResult.Reads.insert({ Offset, Size });
        }
      }
    }
  }

  // Commit results
  HelperCPUStateUsage[&Function] = std::move(HelperResult);
}

GlobalAUAResults
CPUStateUsageAnalysis::collectGlobalAUAResults(const llvm::Function &Function) {
  llvm::Task T({}, "Collecting global Argument Usage Analysis results");
  FastValuePrinter Printer(*Function.getParent());

  GlobalAUAResults Result;
  const aua::Function &FunctionResults = AUA.at(&Function);
  auto &C = TheContext;

  if (Log.isEnabled()) {
    FunctionResults.dump(Log, "");
    Log << DoLog;
  }

  // We are now going to collect interprocedural data for the requested
  // function. This means that we'll integrate the information of the current
  // function "inlining" all of the functions it calls directly or indirectly.

  // If we are going to visit a call with the same context as a previous
  // visit, we'll stop, since it wouldn't provide any additional information.
  // Moreover, in case of recursion, we'll turn the context into FunctionOf
  // the used arguments so that the previous condition will be triggered for
  // sure at the next iteration.

  for (auto &Entry : FunctionResults.localAccesses())
    Result.registerAccess(Entry);

  Result.EscapedArguments = FunctionResults.localEscapedArguments();

  unsigned Iterations = 0;
  std::unordered_set<aua::Call> VisitedCalls;
  struct QueueEntry {
    aua::Call Call;
    llvm::DenseSet<const aua::Function *> FunctionsInStack;
  };
  SmallVector<QueueEntry, 2> Queue;

  // Initialize queue
  for (const aua::Call &Call : FunctionResults.calls())
    Queue.push_back({ Call, {} });

  // TODO: In this loop, we analyze many many times the same function with the
  //       same context.
  //       Some data suggests that we could go from considering 2496029 calls to
  //       just 18014 distinct calls. We'd benefit from having a cache of
  //       analysis results.
  while (not Queue.empty()) {
    ++Iterations;
    const auto &[Current, FunctionsInStack] = Queue.pop_back_val();
    auto &Callee = Current.callee();
    auto &ActualArguments = Current.actualArguments();
    auto SimplifiedActualArguments = ActualArguments;
    auto CalleeName = getCalledFunction(&Current.callInstruction())->getName();
    T.advance(CalleeName);

    for (auto &&[_, SimplifiedActualArgument] : SimplifiedActualArguments) {
      SmallVector<const Value *, 2> Arguments;
      llvm::copy(SimplifiedActualArgument->collect<ArgumentValue>(),
                 std::back_inserter(Arguments));
      SimplifiedActualArgument = &C.getFunctionOf(std::move(Arguments));
    }

    VisitedCalls.insert(Current);

    auto NewFunctionsInStack = FunctionsInStack;
    NewFunctionsInStack.insert(&Callee);

    bool IsRecursiveCall = FunctionsInStack.contains(&Callee);

    if (Log.isEnabled()) {
      Log << "Processing";
      if (IsRecursiveCall)
        Log << " recursive";
      Log << " call:\n";
      Current.dump(Log, "  ", false);
      Log << "  Callee:\n";
      Callee.dump(Log, "    ");
      Log << DoLog;
    }
    LoggerIndent<> Indent(Log);

    // Register escaped arguments replacing arguments
    for (const EscapedArgument &EscapedArgument :
         Callee.localEscapedArguments()) {
      const auto &Adjusted = *C.replaceArguments(C.getArgument(EscapedArgument
                                                                 .index()),
                                                 ActualArguments);
      for (unsigned ArgumentIndex : Adjusted.collectArguments()) {
        // If an argument is not pointing into the tracked data structure,
        // ignore the fact that's escaping
        if (not Initializer.pointsIntoStruct(*Function.getArg(ArgumentIndex)))
          continue;

        bool New = Result.EscapedArguments
                     .insert({ EscapedArgument.location(), ArgumentIndex })
                     .second;
        if (New) {
          revng_log(Log,
                    "Argument "
                      << ArgumentIndex << " of " << Function.getName().str()
                      << " escapes from "
                      << Printer.toString(EscapedArgument.location()) << " in "
                      << EscapedArgument.location().getFunction()->getName());
        }
      }
    }

    // Register memory accesses replacing arguments
    for (const MemoryAccess &Access : Callee.localAccesses()) {
      const auto &Adjusted = *C.replaceArguments(Access.start(),
                                                 ActualArguments);
      MemoryAccess AdjustedAccess = Access.replaceStart(Adjusted);
      revng_log(Log,
                "Registering access " << Access.toString() << " as "
                                      << AdjustedAccess.toString() << " at "
                                      << getName(Access.location()));
      Result.registerAccess(AdjustedAccess);
    }

    // Enqueue calls replacing arguments
    for (const aua::Call &InnerCall : Callee.calls()) {
      auto AdjustedCall = InnerCall;

      const llvm::DenseMap<uint64_t, const Value *>
        *ArgumentsMap = IsRecursiveCall ? &SimplifiedActualArguments :
                                          &ActualArguments;

      for (auto &&[_, ActualArgument] : AdjustedCall.actualArguments()) {
        ActualArgument = C.replaceArguments(*ActualArgument, *ArgumentsMap);
      }

      if (not VisitedCalls.contains(AdjustedCall)) {
        revng_log(Log,
                  "Adding call " << getName(&AdjustedCall.callInstruction()));
        Queue.push_back({ std::move(AdjustedCall), NewFunctionsInStack });
      }
    }
  }

  revng_log(Log, "Collect performed " << Iterations << " iterations");

  return Result;
}

void CPUStateUsageAnalysis::annotate(llvm::Module &M) const {
  // Annotate functions
  for (auto &&[Function, Usage] : HelperCPUStateUsage)
    Annotation(Usage.Escapes, Usage.Reads, Usage.Writes).serialize(*Function);

  // Collect instruction annotations
  std::map<llvm::Instruction *, Annotation> InstructionAnnotations;

  for (auto &&[Use, Access] : MemoryAccessOffsets) {
    auto *I = cast<llvm::Instruction>(Use->getUser());
    if (MemoryAccess::isWrite(Use))
      InstructionAnnotations[I].Writes = Access;
    else
      InstructionAnnotations[I].Reads = Access;
  }

  for (llvm::Instruction *I : Escaping)
    InstructionAnnotations[I].Escapes = true;

  // Annotate instructions
  for (auto &&[ToAnnotate, Annotation] : InstructionAnnotations)
    Annotation.serialize(*ToAnnotate);
}

void StructPointers::visitType(llvm::Type &Type, uint64_t StartingOffset) {
  // Note: this function is recursive but its depth is limited by build time
  //       features, i.e., the depth of the CPU state.
  if (auto *Struct = dyn_cast<llvm::StructType>(&Type)) {
    revng_log(Log,
              "Registering an instance of " << Struct->getName()
                                            << " at offset " << StartingOffset);
    LoggerIndent<> Indent(Log);
    if (Struct->getName().size() != 0)
      OffsetsOfStructs[Struct].push_back(StartingOffset);

    const llvm::StructLayout *Layout = DL.getStructLayout(Struct);
    for (unsigned Index = 0; Index < Struct->getNumElements(); ++Index) {
      visitType(*Struct->getTypeAtIndex(Index),
                StartingOffset + Layout->getElementOffset(Index));
    }
  } else if (auto *Array = dyn_cast<llvm::ArrayType>(&Type)) {
    auto ElementsCount = Array->getNumElements();
    revng_log(Log, "Handling an array of " << ElementsCount << " elements");
    LoggerIndent<> Indent(Log);
    auto &ElementType = *Array->getElementType();
    auto ElementSize = DL.getTypeAllocSize(&ElementType);
    for (unsigned I = 0; I < ElementsCount; ++I)
      visitType(ElementType, StartingOffset + I * ElementSize);
  }
}

void StructPointers::propagateFromActualArguments() {
  bool Again = true;

  while (Again) {
    Again = false;
    SmallVector<std::pair<llvm::Value *, llvm::StructType *>> ToAdd;
    for (auto &&[Value, Struct] : Pointers) {
      auto *Argument = dyn_cast<llvm::Argument>(Value);
      if (Argument == nullptr)
        continue;

      for (llvm::CallBase *Call : callers(Argument->getParent())) {
        auto *V = Call->getArgOperand(Argument->getArgNo());
        if (Pointers.count(V) != 0)
          continue;

        ToAdd.emplace_back(V, Struct);
        Again = true;
      }
    }
    for (auto &&[Value, Struct] : ToAdd)
      Pointers[Value] = Struct;
  }
}

} // namespace aua
