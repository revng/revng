//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <optional>
#include <set>

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/IR/Verifier.h"

#include "revng/ABI/FunctionType/Layout.h"
#include "revng/ABI/ModelHelpers.h"
#include "revng/BasicAnalyses/GeneratedCodeBasicInfo.h"
#include "revng/LocalVariables/LocalVariableBuilder.h"
#include "revng/MFP/MFP.h"
#include "revng/MFP/SetLattices.h"
#include "revng/Model/FunctionTags.h"
#include "revng/Model/IRHelpers.h"
#include "revng/Model/NameBuilder.h"
#include "revng/Model/VerifyHelper.h"
#include "revng/PromoteStackPointer/InstrumentStackAccessesPass.h"
#include "revng/PromoteStackPointer/SegregateStackAccesses.h"
#include "revng/Support/Generator.h"
#include "revng/Support/IRBuilder.h"
#include "revng/Support/IRHelpers.h"
#include "revng/Support/OverflowSafeInt.h"

#include "Helpers.h"

using namespace llvm;
using std::tie;
using StackSpan = abi::FunctionType::Layout::Argument::StackSpan;

static Logger Log("segregate-stack-accesses");

struct OffsetRange {
  int64_t Start = 0;
  int64_t End = 0;
};
using OffsetRanges = SmallVector<OffsetRange>;

static uint64_t getRangeSize(const OffsetRanges &Ranges) {
  uint64_t Result = 0;
  for (const auto &Range : Ranges) {
    Result += Range.End - Range.Start;
  }
  return Result;
}

inline Value *createAdd(revng::IRBuilder &B, Value *V, uint64_t Addend) {
  return B.CreateAdd(V, ConstantInt::get(V->getType(), Addend));
}

inline StringRef stripPrefix(StringRef Prefix, StringRef String) {
  revng_assert(String.startswith(Prefix));
  return String.substr(Prefix.size());
}

inline unsigned getCallPushSize(const model::Binary &Binary) {
  return model::Architecture::getCallPushSize(Binary.Architecture());
}

inline auto snapshot(auto &&Range) {
  SmallVector<std::decay_t<decltype(*Range.begin())>, 16> Result;
  llvm::copy(Range, std::back_inserter(Result));
  return Result;
}

inline unsigned getBitOffsetAt(StructType *Struct, unsigned TargetFieldIndex) {
  unsigned Result = 0;
  for (unsigned FieldIndex = 0; FieldIndex < TargetFieldIndex; ++FieldIndex) {
    Result += Struct->getTypeAtIndex(FieldIndex)->getIntegerBitWidth();
  }
  return Result;
}

inline CallInst *findCallTo(Function *F, Function *ToSearch) {
  CallInst *Call = nullptr;
  for (BasicBlock &BB : *F)
    for (Instruction &I : BB)
      if ((Call = getCallTo(&I, ToSearch)))
        return Call;
  return nullptr;
}

static std::optional<int64_t> getStackOffset(Instruction *I) {
  auto *Pointer = getPointer(I);

  auto *PointerInstruction = dyn_cast<Instruction>(skipCasts(Pointer));
  if (PointerInstruction == nullptr)
    return {};

  if (auto *Call = dyn_cast<CallInst>(PointerInstruction)) {
    if (auto *Callee = getCallee(Call)) {
      if (FunctionTags::StackOffsetMarker.isTagOf(Callee)) {
        // Check if this is a stack access, i.e., targets an exact range
        unsigned AccessSize = getMemoryAccessSize(I);
        auto MaybeStart = getSignedConstantArg(Call, 1);
        auto MaybeEnd = getSignedConstantArg(Call, 2);

        if (MaybeStart and MaybeEnd
            and *MaybeEnd == *MaybeStart + AccessSize + 1) {
          return MaybeStart;
        }
      }
    }
  }

  return {};
}

/// Per-function memoizer for `getStackOffset`. `getStackOffset` walks the
/// def-use chain on every call; this class amortizes that cost by storing
/// the result the first time each instruction is queried.
class StackOffsetCache {
private:
  llvm::DenseMap<llvm::Instruction *, std::optional<int64_t>> Cache;

public:
  std::optional<int64_t> get(llvm::Instruction *I) {
    auto [It, Inserted] = Cache.try_emplace(I);
    if (Inserted)
      It->second = getStackOffset(I);
    return It->second;
  }
};

struct StoredByte {
  int64_t StackOffset = 0;
  llvm::Instruction *Writer = nullptr;

  bool operator<(const StoredByte &Other) const {
    auto ThisTuple = std::tie(StackOffset, Writer);
    auto OtherTuple = std::tie(Other.StackOffset, Other.Writer);
    return ThisTuple < OtherTuple;
  }

  void dump() const debug_function { dump(dbg, 0); }

  template<typename T>
  void dump(T &Stream, unsigned Indent) const {
    for (unsigned I = 0; I < Indent; ++I)
      Stream << "  ";
    Stream << StackOffset << ": " << getName(Writer) << "\n";
  }
};

/// This class tracks the range of stack that's "owned" by a certain alloca.
///
/// You can have multiple instance of this, for instance one for the caller
/// function and then one for each call site.
///
/// In any case, the data in this data structure is always relative to the value
/// of the stack pointer at the entry of the function.
///
/// This means that you will have positive values for stack arguments of the
/// caller, while for all the rest you should expect negative values.
/// Users of this class are expected to follow this rule.
class StackAccessRedirector {
private:
  /// This is for debugging purposes only
  Value *Reference = nullptr;
  std::map<int64_t, std::pair<int64_t, Value *>> Map;

public:
  StackAccessRedirector() = default;
  StackAccessRedirector(Value *Reference) : Reference(Reference) {}

public:
  Value *reference() const { return Reference; }

public:
  void recordSpan(const OffsetRange &Span, Value *BaseAddress) {
    revng_log(Log,
              "Redirecting from " << Span.Start << " to " << Span.End << " to "
                                  << getName(BaseAddress));
    revng_assert(BaseAddress->getType()->isIntegerTy());
    revng_assert(!Map.contains(Span.Start));
    Map[Span.Start] = { Span.End, BaseAddress };

    if (not verify()) {
      dump();
      revng_abort();
    }
    revng_assert(verify());
  }

  void recordSpan(const StackSpan &Span, Value *BaseAddress) {
    OffsetRange NewRange{ static_cast<int64_t>(Span.Offset),
                          static_cast<int64_t>(Span.Offset + Span.Size) };
    return recordSpan(NewRange, BaseAddress);
  }

public:
  std::optional<std::pair<uint64_t, Value *>>
  computeNewBase(int64_t Offset, uint64_t Size) const {

    revng_log(Log, "Searching for " << Offset << " of size " << Size);
    LoggerIndent Indent(Log);

    auto It = Map.upper_bound(Offset);
    if (It == Map.begin()) {
      revng_log(Log, "Not found");
      return std::nullopt;
    }

    --It;

    int64_t SpanStart = It->first;
    int64_t SpanEnd = It->second.first;
    Value *BaseAddress = It->second.second;

    using OSI = OverflowSafeInt<int64_t>;
    auto MaybeEnd = (OSI(Offset) + Size).value();
    if (not MaybeEnd or Offset >= SpanEnd or *MaybeEnd > SpanEnd) {
      revng_log(Log, "Not found");
      return std::nullopt;
    }

    revng_log(Log, "Found");
    return { { Offset - SpanStart, BaseAddress } };
  }

public:
  bool verify() const debug_function {
    if (Map.size() >= 2) {
      auto FirstToSemiLast = llvm::make_range(Map.begin(), --Map.end());
      auto SecondToLast = llvm::make_range(++Map.begin(), Map.end());
      for (auto &&[Current, Next] : llvm::zip(FirstToSemiLast, SecondToLast)) {
        auto CurrentEnd = Current.second.first;
        auto NextStart = Next.first;
        if (CurrentEnd > NextStart)
          return false;
      }
    }

    return true;
  }

  template<typename T>
  void dump(T &Stream) const {
    for (auto &&[K, V] : Map) {
      Stream << K << " -> " << V.first << ": " << getName(V.second) << "\n";
    }
  }

  void dump() const debug_function { dump(dbg); }
};

class FunctionStackAccessRedirectors {
private:
  const StackAccessRedirector *FunctionRedirector = nullptr;
  std::vector<std::unique_ptr<StackAccessRedirector>> CallSiteRedirectors;
  std::map<Instruction *, const StackAccessRedirector *>
    RedirectorForInstruction;

public:
  FunctionStackAccessRedirectors() = default;
  explicit FunctionStackAccessRedirectors(const StackAccessRedirector
                                            *FunctionRedirector) :
    FunctionRedirector(FunctionRedirector) {}

public:
  void setFunctionRedirector(const StackAccessRedirector *Redirector) {
    FunctionRedirector = Redirector;
  }

  bool empty() const {
    return FunctionRedirector == nullptr and CallSiteRedirectors.empty();
  }

  const StackAccessRedirector *get(Instruction *I) const {
    auto It = RedirectorForInstruction.find(I);
    if (It == RedirectorForInstruction.end())
      return FunctionRedirector;
    else
      return It->second;
  }

  const StackAccessRedirector *
  getCommon(const SmallVector<Instruction *> &Instructions) {
    const StackAccessRedirector *Result = FunctionRedirector;
    bool First = true;
    for (auto *Writer : Instructions) {
      const StackAccessRedirector *WriterRedirector = get(Writer);
      if (First) {
        First = false;
        Result = WriterRedirector;
      } else if (WriterRedirector == Result) {
        // All good so far
      } else {
        revng_log(Log,
                  "Different writers of this load are associated to "
                  "different redirectors. Falling back to the "
                  "default one.");
        return FunctionRedirector;
      }
    }

    return Result;
  }

public:
  const StackAccessRedirector *record(StackAccessRedirector &&Redirector) {
    using SAR = StackAccessRedirector;
    CallSiteRedirectors.push_back(std::make_unique<SAR>(std::move(Redirector)));
    return CallSiteRedirectors.back().get();
  }

  void registerOwner(const StackAccessRedirector *Redirector, Instruction *I) {
    revng_log(Log,
              "Registering redirector " << Redirector << " for " << getName(I));
    auto IsRedirector = [&Redirector](const auto &ExistingRedirector) {
      return ExistingRedirector.get() == Redirector;
    };
    revng_assert(llvm::any_of(CallSiteRedirectors, IsRedirector));

    RedirectorForInstruction[I] = Redirector;
  }
};

class MemoryAreaState : public std::set<StoredByte> {
public:
  void eraseRange(int64_t Start, int64_t End) {
    erase(lower_bound(StoredByte{ Start }), upper_bound(StoredByte{ End }));
  }

  void record(int64_t Start, int64_t End, llvm::Instruction *Writer) {
    revng_assert(Start <= End);
    for (auto Offset = Start; Offset < End; ++Offset)
      insert({ Offset, Writer });
  }

public:
  template<typename T>
  void dump(T &Stream, unsigned Indent) const {
    for (const StoredByte &Stored : *this)
      Stored.dump(Stream, Indent);
  }

  void dump() const debug_function { dump(dbg, 0); }
};

class InstructionStackUsage;

class CallSite {
public:
  size_t CallInstructionPushSize = 0;
  std::optional<int64_t> MaybeStackOffset;
  abi::FunctionType::Layout Layout;
  CallInst *OldCall = nullptr;

  OffsetRanges StackArgumentRanges;
  std::optional<OffsetRange> StackReturnValueRange;

  StackAccessRedirector Redirector;

public:
  static CallSite make(CallInst *SSACSCall,
                       const model::TypeDefinition &Prototype,
                       size_t CallInstructionPushSize);

public:
  void processSPTAR(CallInst *InitLocalSPCall,
                    uint64_t StackFrameSize,
                    const MemoryAreaState &State,
                    InstructionStackUsage &StackUsage);

private:
  std::optional<OffsetRange>
  stackArgumentRange(const abi::FunctionType::Layout::Argument &Argument)
    const {

    if (not Argument.Stack.has_value())
      return std::nullopt;

    OverflowSafeInt<int64_t> StackSizeAtCallSite(MaybeStackOffset.value());

    auto StartOffset = StackSizeAtCallSite + Argument.Stack->Offset
                       + CallInstructionPushSize;
    auto EndOffset = StartOffset + Argument.Stack->Size;

    if (not StartOffset or not EndOffset) {
      revng_log(Log,
                "Overflow in computing stack argument offset, "
                "ignoring");
      return std::nullopt;
    }

    return OffsetRange(*StartOffset, *EndOffset);
  }

public:
  template<typename T>
  void dump(T &Stream) const {
    if (MaybeStackOffset.has_value()) {
      Stream << "MaybeStackSize: " << *MaybeStackOffset;
    } else {
      Stream << "MaybeStackSize: no";
    }
    Stream << "\n";

    Stream << "Layout:\n";
    Layout.dump(Stream);
    Stream << "\n";

    Stream << "OldCall: " << getName(OldCall) << "\n";

    Stream << "StackArgumentRanges:\n";
    for (auto [Start, End] : StackArgumentRanges) {
      Stream << "  " << Start << " -> " << End << "\n";
    }

    if (StackReturnValueRange.has_value()) {
      Stream << "StackReturnValueRange: " << StackReturnValueRange->Start
             << ", " << StackReturnValueRange->End;
    } else {
      Stream << "StackReturnValueRange: no";
    }
    Stream << "\n";

    Stream << "Redirector:\n";
    Redirector.dump(Stream);
    Stream << "\n";
  }

  void dump() const debug_function { dump(dbg); }
};

CallSite CallSite::make(CallInst *SSACSCall,
                        const model::TypeDefinition &Prototype,
                        size_t CallInstructionPushSize) {
  CallSite Result;
  Result.Redirector = StackAccessRedirector(SSACSCall);
  Result.CallInstructionPushSize = CallInstructionPushSize;

  // Get stack size at call site
  auto MaybeArgument = getSignedConstantArg(SSACSCall, 0);
  if (MaybeArgument.has_value())
    Result.MaybeStackOffset = -*MaybeArgument;

  // Obtain the prototype layout
  using namespace abi::FunctionType;
  Result.Layout = Layout::make(Prototype);

  // Find old call instruction
  Result.OldCall = findAssociatedCall(SSACSCall);
  revng_assert(Result.OldCall != nullptr);

  if (not Result.MaybeStackOffset.has_value()) {
    revng_log(Log, "Stack size unknown, ignoring stack arguments");
  } else {

    auto &Layout = Result.Layout;

    // Clobber stack arguments
    for (const auto &Argument : Layout.Arguments) {
      if (auto MaybeStackRange = Result.stackArgumentRange(Argument)) {
        Result.StackArgumentRanges.push_back(MaybeStackRange.value());
      }
    }
  }

  return Result;
}

using CallSiteMap = std::map<CallInst *, CallSite>;

/// Given an instruction, returns what parts of the stack clobbers/writes
///
/// For store, we check the pointer operand and store size.
/// For calls, we clobber all the stack arguments and mark as written the stack
/// portion pointed to by the SPTAR, if present.
class InstructionStackUsage {
public:
  struct StackUsage {
    OffsetRanges Clobbers;
    OffsetRanges Writes;
  };

private:
  CallSiteMap &CallSites;
  CallInst *const &InitLocalSPCall;
  const uint64_t &StackFrameSize;
  StackOffsetCache Offsets;

public:
  InstructionStackUsage(CallSiteMap &CallSites,
                        CallInst *const &InitLocalSPCall,
                        const uint64_t &StackFrameSize) :
    CallSites(CallSites),
    InitLocalSPCall(InitLocalSPCall),
    StackFrameSize(StackFrameSize) {}

public:
  /// Returns the (memoized) stack offset of `I`, or `nullopt` if it's not a
  /// stack access. All callers in this pass should query offsets through this
  /// instead of `getStackOffset` directly, so each instruction is analyzed at
  /// most once per function. Non-const because it populates the internal
  /// cache; getReads/getWrites that call it are therefore non-const too.
  std::optional<int64_t> stackOffsetOf(Instruction *I) {
    return Offsets.get(I);
  }

  void detectStackReturnValueRange(Instruction *I,
                                   const MemoryAreaState &State) {
    auto *Call = dyn_cast<CallInst>(I);
    if (Call == nullptr)
      return;

    auto It = CallSites.find(Call);
    if (It == CallSites.end())
      return;

    It->second.processSPTAR(InitLocalSPCall, StackFrameSize, State, *this);
  }

  StackUsage getWrites(Instruction *I) {
    StackUsage Result;

    if (auto *Call = dyn_cast<CallInst>(I)) {
      auto It = CallSites.find(Call);
      if (It == CallSites.end())
        return Result;

      auto &CallSite = It->second;

      for (auto [Start, End] : CallSite.StackArgumentRanges)
        Result.Clobbers.push_back({ Start, End });

      if (CallSite.StackReturnValueRange.has_value()) {
        Result.Writes = { CallSite.StackReturnValueRange.value() };
      }

    } else if (auto *Store = dyn_cast<StoreInst>(I)) {
      // Get stack offset, if available
      auto MaybeStartStackOffset = stackOffsetOf(I);
      if (not MaybeStartStackOffset)
        return Result;

      int64_t StartStackOffset = *MaybeStartStackOffset;
      unsigned AccessSize = getMemoryAccessSize(I);
      int64_t EndStackOffset = StartStackOffset + AccessSize;

      Result.Writes = { { StartStackOffset, EndStackOffset } };
    }

    return Result;
  }

  OffsetRanges getReads(Instruction *I) {
    OffsetRanges Result;

    if (auto *Call = dyn_cast<CallInst>(I)) {
      auto It = CallSites.find(Call);
      if (It == CallSites.end())
        return Result;

      for (auto [Start, End] : It->second.StackArgumentRanges)
        Result.push_back({ Start, End });

    } else if (auto *Store = dyn_cast<LoadInst>(I)) {
      // Get stack offset, if available
      auto MaybeStartStackOffset = stackOffsetOf(I);
      if (not MaybeStartStackOffset)
        return Result;

      int64_t StartStackOffset = *MaybeStartStackOffset;
      unsigned AccessSize = getMemoryAccessSize(I);
      auto EndStackOffset = OverflowSafeInt<int64_t>(StartStackOffset)
                            + AccessSize;

      if (EndStackOffset) {
        return { { StartStackOffset, *EndStackOffset } };
      } else {
        revng_log(Log, "Overflow in StartStackOffset + AccessSize, ignoring");
        return {};
      }
    }

    return Result;
  }
};

[[nodiscard]] static SmallVector<Instruction *>
findAllWriters(const MemoryAreaState &State,
               InstructionStackUsage &StackUsage,
               const OffsetRanges &Ranges) {
  SmallVector<Instruction *> Result;
  DenseMap<Instruction *, unsigned> StoreBytesSeen;

  // `State` is a `std::set<StoredByte>` ordered by `(StackOffset, Writer)`,
  // so iteration yields bytes in ascending offset order. Sort `Ranges` by
  // `Start` and walk both sequences together: for each byte, advance past
  // ranges that already ended; the current range then covers the byte iff
  // its `Start <= Offset`. This turns the previous O(|State| * |Ranges|)
  // double loop into O(|State| + |Ranges| log |Ranges|).
  SmallVector<OffsetRange> SortedRanges(Ranges.begin(), Ranges.end());
  llvm::sort(SortedRanges, [](const OffsetRange &A, const OffsetRange &B) {
    return std::tie(A.Start, A.End) < std::tie(B.Start, B.End);
  });

  size_t RangeIndex = 0;
  for (const StoredByte &Byte : State) {
    int64_t Offset = Byte.StackOffset;

    while (RangeIndex < SortedRanges.size()
           and SortedRanges[RangeIndex].End <= Offset) {
      ++RangeIndex;
    }

    if (RangeIndex == SortedRanges.size())
      break;

    if (SortedRanges[RangeIndex].Start > Offset)
      continue;

    auto It = StoreBytesSeen.find(Byte.Writer);
    if (It == StoreBytesSeen.end()) {
      Result.push_back(Byte.Writer);
      StoreBytesSeen[Byte.Writer] = 1;
    } else {
      ++It->second;
    }
  }

  // Purge entries where the read-write pairs where the read does not fully
  // contain the read or vice-versa, i.e., skip partial overlaps.
  auto RangesSize = getRangeSize(Ranges);
  auto PartiallyOverlaps = [&](Instruction *I) {
    auto OtherRangeSize = getRangeSize(StackUsage.getWrites(I).Writes);
    auto SmallerSize = std::min(RangesSize, OtherRangeSize);
    bool Partial = StoreBytesSeen[I] != SmallerSize;
    if (Partial) {
      revng_log(Log,
                "Dropping "
                  << getName(I) << " due to partial overlap (RangeSize: "
                  << RangesSize << ", OtherRangeSize: " << OtherRangeSize
                  << ", Overlapping: " << StoreBytesSeen[I] << ")");
    }
    return Partial;
  };
  llvm::erase_if(Result, PartiallyOverlaps);

  return Result;
}

void CallSite::processSPTAR(CallInst *InitLocalSPCall,
                            uint64_t StackFrameSize,
                            const MemoryAreaState &State,
                            InstructionStackUsage &StackUsage) {

  if (not Layout.hasSPTAR())
    return;

  revng_log(Log, "Processing SPTAR");
  LoggerIndent Indent(Log);

  if (StackFrameSize == 0) {
    revng_log(Log, "The stack frame has size 0, bailing out");
    return;
  }

  // We expect the first argument to be (revng_undefined_local_sp + constant)

  if (InitLocalSPCall == nullptr) {
    revng_log(Log, "Couldn't find call to revng_undefined_local_sp");
    return;
  }

  revng_assert(Layout.Arguments.size() > 0);
  auto &SPTARArgument = Layout.Arguments[0];
  Value *SPTAR = nullptr;
  if (SPTARArgument.Stack.has_value()) {
    revng_log(Log, "SPTAR is on the stack");

    // The SPTAR is on the stack, we need to try to fetch the only write for
    // that stack slot writing a constant offset from the initial value of the
    // stack pointer.
    revng_assert(SPTARArgument.Registers.size() == 0);
    if (auto MaybeRange = stackArgumentRange(SPTARArgument)) {
      auto Writers = findAllWriters(State, StackUsage, { *MaybeRange });

      // Hopefully there's a single reaching store targeting this slot
      if (Writers.size() == 1 and isa<StoreInst>(Writers[0])) {
        SPTAR = cast<StoreInst>(Writers[0])->getValueOperand();
      } else {
        revng_log(Log,
                  "We were looking for a single store writing the SPTAR "
                  "argument, but failed");
      }
    } else {
      revng_log(Log,
                "Can't obtain the offset range of the SPTAR stack argument");
    }
  } else {
    revng_log(Log, "SPTAR is in a register");
    revng_assert(SPTARArgument.Registers.size() > 0);
    revng_assert(OldCall->arg_size() > 0);
    SPTAR = OldCall->getArgOperand(0);
  }

  using namespace PatternMatch;
  llvm::ConstantInt *Offset = nullptr;
  if (SPTAR == nullptr
      or not match(SPTAR,
                   m_Add(m_Specific(InitLocalSPCall), m_ConstantInt(Offset)))) {
    revng_log(Log,
              "Couldn't identify offset in the stack of the stack-allocated "
              "return value passed via SPTAR");
    return;
  }

  revng_assert(Offset != nullptr);

  OverflowSafeInt<int64_t> EndOffset(Offset->getSExtValue());
  auto ReturnValueSize = Layout.returnValueAggregateType().size();
  revng_assert(ReturnValueSize.has_value());

  EndOffset += ReturnValueSize.value();
  if (EndOffset) {
    // Record the call itself as the writer of the SPTAR range
    StackReturnValueRange = { Offset->getSExtValue(), *EndOffset };
  } else {
    revng_log(Log,
              "Overflow while computing the final offset of "
              "StackReturnValueRange");
  }
}

struct SegregateStackAccessesMFI : public SetUnionLattice<MemoryAreaState> {
public:
  using Label = llvm::BasicBlock *;
  using GraphType = llvm::Function *;
  using ExtraStateKey = llvm::Instruction *;
  using ExtraStateType = mfp::ExtraState<llvm::Instruction *, MemoryAreaState>;

private:
  InstructionStackUsage &StackUsage;

public:
  SegregateStackAccessesMFI(InstructionStackUsage &StackUsage) :
    StackUsage(StackUsage) {}

private:
  void processInstruction(llvm::Instruction &I,
                          MemoryAreaState &StackBytes) const {
    revng_log(Log, "Processing " << getName(&I));
    LoggerIndent Indent(Log);

    StackUsage.detectStackReturnValueRange(&I, StackBytes);

    auto Usage = StackUsage.getWrites(&I);

    for (auto [Start, End] : Usage.Clobbers) {
      revng_log(Log, "Clobbering from " << Start << " to " << End);
      StackBytes.eraseRange(Start, End);
    }

    for (auto [Start, End] : Usage.Writes) {
      revng_log(Log,
                "Recording from " << Start << " to " << End << " as written by "
                                  << getName(&I));
      StackBytes.record(Start, End, &I);
    }
  }

public:
  MemoryAreaState applyTransferFunction(llvm::BasicBlock *BB,
                                        const MemoryAreaState &Value,
                                        ExtraStateType &State) const {
    using namespace llvm;
    revng_log(Log, "Analyzing block " << getName(BB));
    LoggerIndent Indent(Log);

    MemoryAreaState StackBytes = Value;

    // Expose the lattice value at the boundaries of every instruction so
    // the caller can observe the value at specific program points (e.g.
    // the stack_size_at_call_site marker placed right before an isolated
    // call).
    for (Instruction &I : *BB) {
      State.registerBefore(&I, StackBytes);
      processInstruction(I, StackBytes);
      State.registerAfter(&I, StackBytes);
    }

    return StackBytes;
  }
};

static_assert(mfp::MonotoneFrameworkInstance<SegregateStackAccessesMFI>);

namespace mfp {

template<>
void dump<MemoryAreaState>(Logger &Stream,
                           unsigned Indent,
                           const MemoryAreaState &Element) {
  Element.dump(Stream, Indent);
}

template<>
void dumpLabel<BasicBlock *>(Logger &Stream, BasicBlock *const &Label) {
  Stream << getName(Label);
}

template<>
void dumpLabel<Instruction *>(Logger &Stream, Instruction *const &Label) {
  Stream << getName(Label);
}

} // namespace mfp

struct SortByFunction {
  bool operator()(const Instruction *LHS, const Instruction *RHS) const {
    using std::make_pair;
    return make_pair(LHS->getParent(), LHS) < make_pair(RHS->getParent(), RHS);
  }
};

using LVB = LocalVariableBuilder;

static LocalVariableBuilder makeVariableBuilder(const model::Binary &Binary,
                                                llvm::Module &Module) {
  return LVB::make(VariableBuilderTypes(Binary, Module));
}

struct PointersMetadata {
  llvm::SmallVector<bool> ReturnValues;
  llvm::SmallVector<bool> Arguments;
};

static PointersMetadata getPointerMetadata(const abi::FunctionType::Layout &L) {
  PointersMetadata Result;

  for (const auto &R : L.ReturnValues)
    if (L.returnMethod() != abi::FunctionType::ReturnMethod::ModelAggregate)
      Result.ReturnValues.push_back(R.Type->isPointer());

  for (const auto &R : L.Arguments)
    Result.Arguments.push_back(R.Type->isPointer());

  return Result;
}

class SegregateFunctionStack;

/// Rewrite all stack memory accesses
///
/// This pass changes the base address of stack memory access to either:
///
/// * The stack frame of the function (a stack-frame alloca).
/// * The stack arguments of a call site (a call-stack-arguments alloca),
///   which is then passed in as the last argument of the function.
/// * The (newly introduced) last argument of the function representing the
///   stack arguments.
///
/// After this pass, all stack accesses have positive offsets and
/// `revng_undefined_local_sp` is dropped entirely.
///
/// This pass represents local variables as regular LLVM allocas, while
/// accesses are modeled as regular load/store instructions.
class SegregateStackAccesses {
  friend class SegregateFunctionStack;

private:
  const model::Binary &Binary;
  Module &M;
  Function *SSACS = nullptr;
  Function *InitLocalSP = nullptr;
  std::set<Instruction *> ToPurge;
  model::VerifyHelper VH;
  const size_t CallInstructionPushSize = 0;
  std::map<Function *, Function *> OldToNew;
  SmallVector<Instruction *> ToPushALAP;

  llvm::Type *TargetPointerSizedInteger = nullptr;
  llvm::Type *OpaquePointerType = nullptr;
  LocalVariableBuilder VariableBuilder;

public:
  SegregateStackAccesses(const model::Binary &Binary, llvm::Module &M) :
    Binary(Binary),
    M(M),
    SSACS(getIRHelper("stack_size_at_call_site", M)),
    InitLocalSP(getIRHelper("revng_undefined_local_sp", M)),
    CallInstructionPushSize(getCallPushSize(Binary)),
    TargetPointerSizedInteger(getPointerSizedInteger(M.getContext(),
                                                     Binary.Architecture())),
    OpaquePointerType(PointerType::get(M.getContext(), 0)),
    VariableBuilder(makeVariableBuilder(Binary, M)) {}

public:
  bool prologue() {
    upgradeDynamicFunctions();
    return true;
  }

  bool runOnFunction(const model::Function &ModelFunction,
                     llvm::Function &Function);

  bool epilogue() {
    pushALAP();

    // Purge stores that have been used at least once
    for (Instruction *I : ToPurge)
      eraseFromParent(I);

    // Erase original functions
    for (auto &&[OldFunction, NewFunction] : OldToNew)
      eraseFromParent(OldFunction);

    return true;
  }

private:
  void upgradeDynamicFunctions() {
    SmallVector<Function *, 8> Functions;
    for (Function &F : FunctionTags::DynamicFunction.functions(&M))
      Functions.push_back(&F);

    // Identify all functions that have stack arguments
    for (Function *OldFunction : Functions) {
      // TODO: this is not very nice
      auto SymbolName = stripPrefix("dynamic_", OldFunction->getName()).str();
      auto &ImportedFunction = Binary.ImportedDynamicFunctions().at(SymbolName);
      auto &ProtoT = *Binary.prototypeOrDefault(ImportedFunction.prototype());
      recreateApplyingModelPrototype(OldFunction, ProtoT);
    }
  }

  std::pair<llvm::Function *, abi::FunctionType::Layout>
  getOrCreateNewLocalFunction(Function *OldFunction) {
    MetaAddress Entry = getMetaAddressMetadata(OldFunction,
                                               "revng.function.entry");
    revng_assert(Entry.isValid());

    const model::Function &ModelFunction = Binary.Functions().at(Entry);

    // Create new FunctionType
    auto &Prototype = *Binary.prototypeOrDefault(ModelFunction.prototype());

    return recreateApplyingModelPrototype(OldFunction, Prototype);
  }

  llvm::Function *getOrCreateNewFunction(Function *OldFunction) {
    MetaAddress Entry = getMetaAddressMetadata(OldFunction,
                                               "revng.function.entry");

    if (Entry.isValid()) {
      return getOrCreateNewLocalFunction(OldFunction).first;
    } else {
      revng_assert(FunctionTags::DynamicFunction.isTagOf(OldFunction));
      return OldToNew.at(OldFunction);
    }
  }

  void pushALAP() {
    // Push ALAP all stack arguments allocations
    Function *LastFunction = nullptr;
    DominatorTree DT;
    for (Instruction *I : ToPushALAP) {
      if (not I->getNumUses())
        continue;
      Function *F = I->getParent()->getParent();
      if (F != LastFunction) {
        LastFunction = F;
        DT.recalculate(*LastFunction);
      }

      pushInstructionALAP(DT, I);
    }
  }

private:
  std::pair<llvm::Function *, abi::FunctionType::Layout>
  recreateApplyingModelPrototype(Function *OldFunction,
                                 const model::TypeDefinition &Prototype) {
    using namespace abi::FunctionType;
    auto Layout = Layout::make(Prototype);

    LLVMContext &Context = OldFunction->getContext();
    auto Architecture = Binary.Architecture();

    Type *OldReturnType = OldFunction->getReturnType();
    FunctionType &NewType = layoutToLLVMFunctionType<false>(Context,
                                                            Architecture,
                                                            Layout);

    // NOTE: all the model *must* be read above this line!
    //       If we don't do this, we will break invalidation tracking
    //       information.
    auto It = OldToNew.find(OldFunction);
    if (It != OldToNew.end())
      return { It->second, Layout };

    // Create the new function, stealing the name
    Function &NewFunction = recreateWithoutBody(*OldFunction, NewType);
    const auto &[PointerReturns, PointerArguments] = getPointerMetadata(Layout);
    setPointersMetadata(&NewFunction, PointerReturns, PointerArguments);

    // Record the old-to-new mapping
    OldToNew[OldFunction] = &NewFunction;

    return { &NewFunction, Layout };
  }
};

/// Per-function state and logic for SegregateStackAccesses. One instance is
/// constructed per call to runOnFunction. `upgrade()` rewrites the function
/// signature and lowers arguments/return values; `segregate()` then runs the
/// data-flow analysis and redirects every stack access to its proper base.
class SegregateFunctionStack {
private:
  SegregateStackAccesses &SSA;
  const model::Function &ModelFunction;
  llvm::Function &OldFunction;

  // Populated by upgrade()
  llvm::Function *NewFunction = nullptr;
  abi::FunctionType::Layout Layout;
  std::map<model::Register::Values, llvm::Argument *> ArgumentToRegister;
  StackAccessRedirector StackArgumentRedirector;
  bool HasStackArgumentRedirector = false;
  llvm::Value *ReturnValueAllocation = nullptr;
  llvm::Value *ReturnValueIntAddress = nullptr;

  // Populated by segregate()
  llvm::CallInst *InitLocalSPCall = nullptr;
  uint64_t StackFrameSize = 0;
  CallSiteMap CallSites;
  SegregateStackAccessesMFI::ExtraStateType MFPExtraState;
  InstructionStackUsage StackUsage;
  FunctionStackAccessRedirectors Redirectors;

public:
  SegregateFunctionStack(SegregateStackAccesses &P,
                         const model::Function &ModelFunction,
                         llvm::Function &OldFunction) :
    SSA(P),
    ModelFunction(ModelFunction),
    OldFunction(OldFunction),
    StackUsage(CallSites, InitLocalSPCall, StackFrameSize) {}

public:
  void upgrade();
  void segregate();

private:
  void setupNewFunction();
  void checkReturnMethod();
  void prepareReturnValueStorage();
  void lowerArguments(revng::IRBuilder &B);
  void lowerReturnValues(revng::IRBuilder &B);

  /// Record a stack argument's storage in the function-level redirector.
  ///
  /// Layout sketch:
  ///
  ///   0x0000
  ///                 -16
  ///     _________   -8
  ///    |_________|  +0  Saved return address
  ///    |_________|  +8  struct StackArguments { uint64_t Offset0;
  ///    |_________|  +16   uint64_t Offset8; };
  ///    |_________|  +24
  ///   _|_________|_
  ///
  ///   0xffff
  void recordFunctionStackArgument(const StackSpan &Span, llvm::Value *V) {
    StackArgumentRedirector.recordSpan(SSA.CallInstructionPushSize + Span, V);
  }

  void initialize();
  void collectCallSites();
  void runDataFlowAnalysis();
  void lowerCallSites();
  void redirectMemoryAccesses();
  void adjustStackFrame();
  StackAccessRedirector handleCallSite(llvm::CallInst *SSACSCall,
                                       const CallSite &CallSite);

  void handleMemoryAccess(const StackAccessRedirector &Redirector,
                          llvm::Instruction *I,
                          std::optional<int64_t> MaybeStackOffset) {
    if (not MaybeStackOffset)
      return;
    int64_t StackOffset = *MaybeStackOffset;
    revng_log(Log, "StackOffset: " << StackOffset);

    unsigned AccessSize = getMemoryAccessSize(I);
    auto NewBase = Redirector.computeNewBase(StackOffset, AccessSize);
    if (NewBase)
      replace(I, NewBase->second, NewBase->first);
  }

  llvm::Value *pointer(revng::IRBuilder &B, llvm::Value *V) const {
    return B.CreateIntToPtr(V, SSA.OpaquePointerType);
  }

  llvm::Value *
  computeAddress(revng::IRBuilder &B, llvm::Value *Base, int64_t Offset) const {
    return pointer(B, createAdd(B, Base, Offset));
  }

  void replace(llvm::Instruction *I, llvm::Value *Base, int64_t Offset) {
    SSA.ToPurge.insert(I);

    revng::IRBuilder B(I);
    auto *NewAddress = computeAddress(B, Base, Offset);

    llvm::Instruction *NewInstruction = nullptr;
    if (auto *Store = dyn_cast<llvm::StoreInst>(I)) {
      NewInstruction = B.CreateStore(Store->getValueOperand(), NewAddress);
    } else if (auto *Load = dyn_cast<llvm::LoadInst>(I)) {
      NewInstruction = B.CreateLoad(I->getType(), NewAddress);
    }

    I->replaceAllUsesWith(NewInstruction);
    NewInstruction->copyMetadata(*I);
  }

  llvm::Constant *getSPConstant(uint64_t Value) const {
    return llvm::ConstantInt::get(SSA.TargetPointerSizedInteger, Value);
  }

  unsigned
  shiftAmount(unsigned Offset, unsigned NewSize, unsigned OldSize) const {
    if (NewSize >= OldSize)
      return 0;
    if (model::Architecture::isLittleEndian(SSA.Binary.Architecture())) {
      return Offset * 8;
    } else {
      return (NewSize - Offset - OldSize) * 8;
    }
  }
};

void SegregateFunctionStack::upgrade() {
  revng_log(Log, "Upgrading " << getName(&OldFunction));
  LoggerIndent Indent(Log);

  setupNewFunction();

  checkReturnMethod();

  prepareReturnValueStorage();

  revng::IRBuilder B(NewFunction->getContext());
  B.setInsertPointToFirstNonAlloca(*NewFunction);

  lowerArguments(B);

  lowerReturnValues(B);

  for (BasicBlock &BB : *NewFunction) {
    revng_assert(BB.getTerminator() != nullptr);
  }
}

void SegregateFunctionStack::segregate() {
  revng_log(Log,
            "Segregating "
              << model::CNameBuilder(SSA.Binary).name(ModelFunction));
  LoggerIndent Indent(Log);

  initialize();

  collectCallSites();

  runDataFlowAnalysis();

  lowerCallSites();

  redirectMemoryAccesses();

  adjustStackFrame();
}

void SegregateFunctionStack::setupNewFunction() {
  auto &&[NewFn, NewLayout] = SSA.getOrCreateNewLocalFunction(&OldFunction);
  NewFunction = NewFn;
  Layout = std::move(NewLayout);

  // Let the new function steal the body from the old function
  moveBlocksInto(OldFunction, *NewFunction);

  FunctionTags::StackAccessesSegregated.addTo(NewFunction);

  // Map llvm::Argument * to model::Register
  auto ArgumentRegisters = Layout.argumentRegisters();
  for (const auto &[Register, OldArgument] :
       zip(ArgumentRegisters, OldFunction.args())) {
    ArgumentToRegister[Register] = &OldArgument;
  }

  // Decide whether we need a redirector for the function's stack arguments
  auto IsStackArgument = [](const auto &Argument) -> bool {
    return Argument.Stack.has_value();
  };
  if (llvm::any_of(Layout.Arguments, IsStackArgument)) {
    revng_log(Log, "Creating redirector for stack arguments");
    HasStackArgumentRedirector = true;
  }
}

void SegregateFunctionStack::checkReturnMethod() {
  using namespace abi::FunctionType;

  Type *NewReturnType = NewFunction->getReturnType();
  switch (Layout.returnMethod()) {
  case ReturnMethod::Void:
    revng_assert(NewReturnType->isVoidTy());
    break;

  case ReturnMethod::ModelAggregate:
    if (Layout.hasSPTAR()) {
      // Nothing to check here
    } else {
      revng_assert(NewReturnType->isArrayTy());
      auto *ArrayTy = cast<llvm::ArrayType>(NewReturnType);
      auto *ElemTy = ArrayTy->getElementType();
      revng_assert(cast<llvm::IntegerType>(ElemTy)->getBitWidth() == 8);
      unsigned NumElems = ArrayTy->getNumElements();
      size_t ModelAggregateSize = *Layout.returnValueAggregateType().size();
      revng_assert(ModelAggregateSize == NumElems);
    }
    break;

  case ReturnMethod::RegisterSet:
    // Assert each return instruction is using a StructInitializer
    for (BasicBlock &BB : *NewFunction) {
      if (auto *Ret = dyn_cast<ReturnInst>(BB.getTerminator())) {
        auto *Call = cast<CallInst>(Ret->getReturnValue());
        auto *Callee = getCalledFunction(Call);
        revng_assert(Call != nullptr);
        revng_assert(FunctionTags::StructInitializer.isTagOf(Callee));
      }
    }
    break;

  case ReturnMethod::Scalar:
    break;
  }
}

void SegregateFunctionStack::prepareReturnValueStorage() {
  using namespace abi::FunctionType;

  if (Layout.returnMethod() != ReturnMethod::ModelAggregate)
    return;

  const model::Type &A = Layout.returnValueAggregateType();
  SSA.VariableBuilder.setTargetFunction(NewFunction);
  tie(ReturnValueAllocation,
      ReturnValueIntAddress) = SSA.VariableBuilder
                                 .createLocalVariableAndTakeIntAddress(A);
  revng_assert(ReturnValueAllocation);
  revng_assert(ReturnValueIntAddress);

  if (not Layout.hasSPTAR())
    return;

  // Identify the SPTAR and redirect the argument pointing at the storage of
  // the return value to our freshly created local variable.
  auto &ModelArgument = Layout.Arguments[0];
  if (ModelArgument.Stack) {
    revng_assert(ModelArgument.Registers.size() == 0);
    recordFunctionStackArgument(*ModelArgument.Stack, ReturnValueIntAddress);
  } else {
    revng_assert(ModelArgument.Registers.size() == 1);
    Argument *OldArgument = ArgumentToRegister.at(ModelArgument.Registers[0]);
    OldArgument->replaceAllUsesWith(ReturnValueIntAddress);
  }
}

void SegregateFunctionStack::lowerArguments(revng::IRBuilder &B) {
  using namespace abi::FunctionType;
  using namespace abi::FunctionType::ArgumentKind;

  // The SPTAR argument was handled in prepareReturnValueStorage; drop it
  // from the list of arguments we still need to lower.
  auto ModelArguments = llvm::make_range(Layout.Arguments.begin(),
                                         Layout.Arguments.end());
  if (Layout.returnMethod() == ReturnMethod::ModelAggregate
      and Layout.hasSPTAR()) {
    ModelArguments = llvm::drop_begin(ModelArguments);
  }

  for (auto &&[ModelArgument, NewArgument] :
       zip(ModelArguments, NewFunction->args())) {

    unsigned OffsetInNewArgument = 0;
    Type *NewArgumentType = NewArgument.getType();
    unsigned NewArgumentSize = NewArgumentType->getIntegerBitWidth() / 8;

    llvm::Value *ToRecordSpan = nullptr;
    bool UsesStack = ModelArgument.Stack.has_value();

    if (ModelArgument.Kind == PointerToCopy) {
      auto Architecture = SSA.Binary.Architecture();
      auto PointerSize = model::Architecture::getPointerSize(Architecture);
      revng_assert(ModelArgument.Type->size() > PointerSize);
      Value *AddressOfNewArgument = &NewArgument;

      if (UsesStack) {
        // When loading from this stack slot, return the address of the
        // address of the new argument
        revng_assert(ModelArgument.Registers.size() == 0);
        ToRecordSpan = AddressOfNewArgument;
      } else {
        // Replace the old argument with an address of the new argument
        revng_assert(ModelArgument.Registers.size() == 1);
        auto Register = ModelArgument.Registers[0];
        Argument *OldArgument = ArgumentToRegister.at(Register);
        OldArgument->replaceAllUsesWith(AddressOfNewArgument);
      }

    } else if (ModelArgument.Kind == Scalar) {
      revng_assert(ModelArgument.Type->isScalar());
      for (model::Register::Values Register : ModelArgument.Registers) {
        Argument *OldArgument = ArgumentToRegister.at(Register);
        Type *OldArgumentType = OldArgument->getType();
        auto OldArgumentSize = OldArgumentType->getIntegerBitWidth() / 8;
        revng_assert(model::Register::getSize(Register) == OldArgumentSize);

        unsigned ShiftAmount = shiftAmount(OffsetInNewArgument,
                                           NewArgumentSize,
                                           OldArgumentSize);

        Value *Shifted = &NewArgument;
        if (ShiftAmount != 0)
          Shifted = B.CreateLShr(&NewArgument, ShiftAmount);
        Value *Trunced = B.CreateZExtOrTrunc(Shifted, OldArgumentType);

        OldArgument->replaceAllUsesWith(Trunced);

        OffsetInNewArgument += OldArgumentSize;
      }

      if (ModelArgument.Stack) {
        Type *ArgumentType = NewArgument.getType();
        auto Pair = SSA.VariableBuilder.createAllocaWithPtrToInt(NewFunction,
                                                                 ArgumentType);
        auto [Alloca, PtrToInt] = Pair;
        B.CreateStore(&NewArgument, Alloca);
        ToRecordSpan = PtrToInt;
      }

    } else if (ModelArgument.Kind == ReferenceToAggregate) {
      Value *AddressOfNewArgument = &NewArgument;

      for (model::Register::Values Register : ModelArgument.Registers) {
        Argument *OldArgument = ArgumentToRegister.at(Register);

        Value *ArgumentPointer = computeAddress(B,
                                                AddressOfNewArgument,
                                                OffsetInNewArgument);
        Value *ArgumentValue = B.CreateLoad(OldArgument->getType(),
                                            ArgumentPointer);

        OldArgument->replaceAllUsesWith(ArgumentValue);

        OffsetInNewArgument += model::Register::getSize(Register);
      }

      if (ModelArgument.Stack)
        ToRecordSpan = AddressOfNewArgument;
    }

    if (ToRecordSpan) {
      recordFunctionStackArgument(*ModelArgument.Stack, ToRecordSpan);
    }
  }
}

void SegregateFunctionStack::lowerReturnValues(revng::IRBuilder &B) {
  using namespace abi::FunctionType;

  Type *NewReturnType = NewFunction->getReturnType();

  llvm::SmallVector<ReturnInst *, 4> Returns;
  for (BasicBlock &BB : *NewFunction) {
    if (auto *Ret = dyn_cast<ReturnInst>(BB.getTerminator()))
      Returns.push_back(Ret);
  }

  for (BasicBlock &BB : *NewFunction) {
    revng_assert(BB.getTerminator() != nullptr);
  }

  switch (Layout.returnMethod()) {
  case ReturnMethod::ModelAggregate: {
    // Replace return instructions with returning a copy of the local variable
    // representing the return value
    for (ReturnInst *Ret : Returns) {
      B.SetInsertPoint(Ret);

      if (not Layout.hasSPTAR()) {
        // We have an aggregate returned through registers, fill in the
        // struct using stores
        revng_assert(Layout.returnValueRegisterCount() > 0);

        llvm::SmallVector<llvm::Value *, 4> ReturnValues;
        Value *RetValue = Ret->getReturnValue();
        if (Layout.returnValueRegisterCount() == 1) {
          ReturnValues.push_back(RetValue);
        } else {
          auto *Call = cast<CallInst>(Ret->getReturnValue());
          auto *Callee = getCalledFunction(Call);
          revng_assert(Call != nullptr);
          revng_assert(FunctionTags::StructInitializer.isTagOf(Callee));
          llvm::copy(Call->args(), std::back_inserter(ReturnValues));
        }

        uint64_t Offset = 0;
        for (Value *ReturnValue : ReturnValues) {
          Value *Pointer = createAdd(B, ReturnValueIntAddress, Offset);
          B.CreateStore(ReturnValue, pointer(B, Pointer));
          Offset += ReturnValue->getType()->getIntegerBitWidth() / 8;
        }
      }

      revng_assert(ReturnValueAllocation);
      revng_assert(ReturnValueIntAddress);
      // TODO: we should review all the CreateLoad alignments
      Value *ToReturn = B.CreateLoad(NewReturnType, ReturnValueAllocation);
      B.CreateRet(ToReturn);
      Ret->eraseFromParent();
    }
  } break;

  case ReturnMethod::Scalar: {
    Type *OldReturnType = OldFunction.getReturnType();

    if (OldReturnType != NewReturnType) {
      if (OldReturnType->isIntegerTy() and NewReturnType->isIntegerTy()) {
        // Handle return values smaller than the original function
        for (ReturnInst *Ret : Returns) {
          B.SetInsertPoint(Ret);
          B.CreateRet(B.CreateTrunc(Ret->getReturnValue(), NewReturnType));
          Ret->eraseFromParent();
        }
      } else if (OldReturnType->isStructTy() and NewReturnType->isIntegerTy()) {
        // Turn struct_initializer into an integer
        for (ReturnInst *Ret : Returns) {
          auto *Call = cast<CallInst>(Ret->getReturnValue());
          auto *Callee = getCalledFunction(Call);
          revng_assert(Call != nullptr);
          revng_assert(FunctionTags::StructInitializer.isTagOf(Callee));

          B.SetInsertPoint(Ret);
          Value *Accumulator = ConstantInt::get(NewReturnType, 0);
          uint64_t ShiftAmount = 0;
          for (Value *Argument : Call->args()) {
            auto *Extended = B.CreateZExtOrTrunc(Argument, NewReturnType);
            Accumulator = B.CreateOr(Accumulator,
                                     B.CreateShl(Extended, ShiftAmount));
            ShiftAmount += Argument->getType()->getIntegerBitWidth();
          }
          B.CreateRet(Accumulator);
          Ret->eraseFromParent();
          Call->eraseFromParent();
        }
      }
    }
  } break;

  case ReturnMethod::Void:
  case ReturnMethod::RegisterSet:
    break;

  default:
    revng_abort();
  }
}

void SegregateFunctionStack::initialize() {
  if (SSA.InitLocalSP != nullptr)
    InitLocalSPCall = findCallTo(NewFunction, SSA.InitLocalSP);

  if (const model::TypeDefinition *T = ModelFunction.stackFrameType())
    StackFrameSize = *rc_eval(T->size(SSA.VH));

  revng_log(Log, "StackFrameSize: " << StackFrameSize);

  StackAccessRedirector *Default = HasStackArgumentRedirector ?
                                     &StackArgumentRedirector :
                                     nullptr;
  revng_log(Log, "Default redirector: " << (Default ? "yes" : "no"));
  Redirectors.setFunctionRedirector(Default);
}

void SegregateFunctionStack::collectCallSites() {
  for (BasicBlock &BB : *NewFunction) {
    for (Instruction &I : BB) {
      CallInst *SSACSCall = nullptr;
      if (SSA.SSACS != nullptr and (SSACSCall = getCallTo(&I, SSA.SSACS))) {
        revng_log(Log, "Processing " << getName(SSACSCall));
        LoggerIndent Indent(Log);

        MFPExtraState.registerAsInterestingBefore(SSACSCall);

        const auto &Prototype = *getCallSitePrototype(SSA.Binary, SSACSCall);
        CallSites[SSACSCall] = CallSite::make(SSACSCall,
                                              Prototype,
                                              SSA.CallInstructionPushSize);

        if (Log.isEnabled()) {
          CallSites[SSACSCall].dump(Log);
          Log << DoLog;
        }

      } else if (isa<LoadInst>(&I)
                 and StackUsage.stackOffsetOf(&I).has_value()) {
        // Handle load from the stack
        revng_log(Log, "Registering " << getName(&I) << " as interesting");
        MFPExtraState.registerAsInterestingBefore(&I);
      }
    }
  }
}

void SegregateFunctionStack::runDataFlowAnalysis() {
  revng_log(Log, "Running SegregateStackAccessesMFI");
  LoggerIndent Indent(Log);
  BasicBlock *Entry = &NewFunction->getEntryBlock();
  std::vector ExtremalLabels = { Entry };
  using SSAMFI = SegregateStackAccessesMFI;
  SSAMFI MFI(StackUsage);
  mfp::getMaximalFixedPoint<SSAMFI>({ .Instance = &MFI,
                                      .Flow = NewFunction,
                                      .ExtremalLabels = &ExtremalLabels,
                                      .ExtraState = &MFPExtraState,
                                      .Logger = &Log });

  if (Log.isEnabled()) {
    Log << "Extra state:\n";
    MFPExtraState.dump(Log);
    Log << DoLog;
  }
}

void SegregateFunctionStack::lowerCallSites() {
  revng_log(Log, "Handling call sites");
  LoggerIndent Indent(Log);
  for (auto &[SSACSCall, CallSite] : CallSites) {
    revng_log(Log, "Handling " << getName(SSACSCall));
    LoggerIndent Indent(Log);

    const auto &AnalysisResult = MFPExtraState.getBefore(SSACSCall);

    auto *CallSiteRedirector = Redirectors.record(handleCallSite(SSACSCall,
                                                                 CallSite));
    Redirectors.registerOwner(CallSiteRedirector, SSACSCall);

    if (Log.isEnabled()) {
      Log << "Status of the analysis at call site:\n";
      AnalysisResult.dump(Log, 1);
      Log << DoLog;
    }

    for (Instruction *I : findAllWriters(AnalysisResult,
                                         StackUsage,
                                         CallSite.StackArgumentRanges)) {
      Redirectors.registerOwner(CallSiteRedirector, I);
    }
  }
}

void SegregateFunctionStack::redirectMemoryAccesses() {
  if (Redirectors.empty())
    return;

  revng_log(Log, "Handling memory accesses");
  LoggerIndent Indent(Log);
  for (BasicBlock &BB : *NewFunction) {
    for (Instruction &I : BB) {
      if (not(isa<LoadInst>(&I) or isa<StoreInst>(&I)))
        continue;

      if (isa<LoadInst>(&I))
        revng_log(Log, "Handling load " << getName(&I));
      else
        revng_log(Log, "Handling store " << getName(&I));
      LoggerIndent Indent(Log);

      auto MaybeStackOffset = StackUsage.stackOffsetOf(&I);
      if (Log.isEnabled()) {
        Log << "StackOffset: ";
        if (MaybeStackOffset.has_value())
          Log << *MaybeStackOffset;
        else
          Log << "(none)";
        Log << DoLog;
      }

      // Find the correct redirector
      const StackAccessRedirector *Redirector = nullptr;
      if (isa<LoadInst>(&I) and MaybeStackOffset.has_value()) {
        // For loads, choose the redirector of its writers
        const auto &AnalysisResult = MFPExtraState.getBefore(&I);

        auto ReadRanges = StackUsage.getReads(&I);
        auto Writers = findAllWriters(AnalysisResult, StackUsage, ReadRanges);

        if (Log.isEnabled()) {
          Log << "Found " << Writers.size() << " writers:\n";
          for (auto *Writer : Writers) {
            Log << "  " << getName(Writer) << "\n";
          }
          Log << DoLog;
        }

        Redirector = Redirectors.getCommon(Writers);
      } else {
        Redirector = Redirectors.get(&I);
      }

      if (Redirector == nullptr) {
        revng_log(Log, "No redirector");
      } else {
        revng_log(Log,
                  "Redirecting using " << getName(Redirector->reference()));
        handleMemoryAccess(*Redirector, &I, MaybeStackOffset);
      }
    }
  }
}

void SegregateFunctionStack::adjustStackFrame() {
  if (InitLocalSPCall == nullptr or ModelFunction.StackFrame().Type().isEmpty())
    return;

  // Create call and rebase SP0, if StackFrameSize is not zero
  if (StackFrameSize == 0)
    return;

  model::UpcastableType FrameType = ModelFunction.StackFrame().Type();
  SSA.VariableBuilder.setTargetFunction(NewFunction);
  Instruction *StackFrameAddress = SSA.VariableBuilder
                                     .createStackFrameVariable(FrameType);

  revng::IRBuilder Builder(InitLocalSPCall);
  auto *SP0 = Builder.CreateAdd(StackFrameAddress,
                                getSPConstant(StackFrameSize));
  InitLocalSPCall->replaceAllUsesWith(SP0);

  // Cleanup revng_undefined_local_sp
  eraseFromParent(InitLocalSPCall);
}

StackAccessRedirector
SegregateFunctionStack::handleCallSite(llvm::CallInst *SSACSCall,
                                       const CallSite &CallSite) {
  using namespace abi::FunctionType;

  revng_log(Log, "Analyzing call to SSACS " << getName(SSACSCall));
  LoggerIndent Indent(Log);

  Function *Caller = SSACSCall->getParent()->getParent();

  // Unpack CallSite
  const auto
    &[_1, MaybeStackOffsetAtCallSite, Layout, OldCall, _2, _3, _4] = CallSite;

  revng::IRBuilder B(OldCall);

  // Map llvm::Argument * to model::Register
  std::map<model::Register::Values, llvm::Value *> ArgumentToRegister;
  auto ArgumentRegisters = Layout.argumentRegisters();
  for (auto &&[Register, OldArg] : zip(ArgumentRegisters, OldCall->args()))
    ArgumentToRegister[Register] = OldArg.get();

  // Check if it's a direct call
  auto *Callee = dyn_cast<Function>(OldCall->getCalledOperand());
  bool IsDirect = (Callee != nullptr);

  // Obtain or compute the function type for the call
  FunctionType *CalleeType = nullptr;
  Value *CalledValue = nullptr;
  if (IsDirect) {
    Function *NewCallee = SSA.getOrCreateNewFunction(Callee);
    CalledValue = NewCallee;
    CalleeType = NewCallee->getFunctionType();
  } else {
    LLVMContext &Context = OldCall->getContext();
    auto Architecture = SSA.Binary.Architecture();
    CalleeType = &layoutToLLVMFunctionType<false>(Context,
                                                  Architecture,
                                                  Layout);
    CalledValue = B.CreateBitCast(OldCall->getCalledOperand(),
                                  CalleeType->getPointerTo());
  }

  SmallVector<llvm::Value *, 4> Arguments;

  StackAccessRedirector Redirector;
  auto RecordStackArgument =
    [this, &Redirector, &MaybeStackOffsetAtCallSite](const StackSpan &StackSpan,
                                                     Value *V) {
      revng_assert(MaybeStackOffsetAtCallSite);
      // Record its portion of the stack for redirection

      // 0x0000
      // _____________ -40
      //  |_________|  -32 Saved return address, MaybeStackSize
      //  |_________|  -24 struct StackArguments { uint64_t Offset0;
      //  |_________|  -16  uint64_t Offset8; };
      // _|_________|_ -8  Local variable
      //  |_________|  +0  Saved return address
      // _|_________|_
      //
      // 0xffff

      Redirector.recordSpan(*MaybeStackOffsetAtCallSite
                              + SSA.CallInstructionPushSize + StackSpan,
                            V);
    };

  SmallVector<llvm::Type *, 8> LLVMArgumentTypes;
  bool HasSPTAR = Layout.hasSPTAR();

  auto ReturnMethod = Layout.returnMethod();
  if (HasSPTAR) {
    revng_log(Log, "This call site has a SPTAR");
    revng_assert(ReturnMethod == ReturnMethod::ModelAggregate);

    // The original function produced by enforce-abi had the SPTAR but the
    // re-created one doesn't, re-inject it temporarily
    revng_assert(Layout.Arguments.size() > 0);
    uint64_t SPTARSize = *Layout.Arguments[0].Type->size();
    LLVMArgumentTypes.push_back(B.getIntNTy(SPTARSize * 8));
  }

  copy(CalleeType->params(), std::back_inserter(LLVMArgumentTypes));

  bool MessageEmitted = false;
  for (auto &&[LLVMType, ModelArgument] :
       llvm::zip(LLVMArgumentTypes, Layout.Arguments)) {
    uint64_t NewSize = *ModelArgument.Type->size();

    switch (ModelArgument.Kind) {

    case ArgumentKind::PointerToCopy: {
      Value *Pointer = nullptr;

      if (ModelArgument.Stack) {
        model::Architecture::Values Architecture = SSA.Binary.Architecture();
        auto PointerSize = model::Architecture::getPointerSize(Architecture);
        revng_assert(ModelArgument.Type->size() > PointerSize);
        revng_assert(ModelArgument.Registers.size() == 0);
        revng_assert(ModelArgument.Stack->Size == PointerSize);
        revng_assert(MaybeStackOffsetAtCallSite);

        // Create an alloca
        auto *StackSpanType = B.getIntNTy(ModelArgument.Stack->Size * 8);
        auto Pair = SSA.VariableBuilder.createAllocaWithPtrToInt(Caller,
                                                                 StackSpanType);
        auto [Alloca, PtrToInt] = Pair;

        RecordStackArgument(*ModelArgument.Stack, PtrToInt);

        // Load the alloca and record it as a pointer
        Pointer = B.CreateLoad(Alloca->getAllocatedType(), Alloca);
      } else {
        revng_assert(ModelArgument.Registers.size() == 1);
        auto Register = ModelArgument.Registers[0];
        Pointer = ArgumentToRegister.at(Register);
      }

      // Pass as argument the pointer computed above.
      Arguments.push_back(Pointer);
    } break;

    case ArgumentKind::Scalar:
    case ArgumentKind::ShadowPointerToAggregateReturnValue: {
      revng_assert(ModelArgument.Type->isScalar());
      Value *Accumulator = ConstantInt::get(LLVMType, 0);
      unsigned OffsetInNewArgument = 0;
      for (auto &Register : ModelArgument.Registers) {
        Value *OldArgument = ArgumentToRegister.at(Register);
        unsigned OldSize = model::Register::getSize(Register);

        Value *Extended = B.CreateZExtOrTrunc(OldArgument, LLVMType);

        unsigned ShiftAmount = shiftAmount(OffsetInNewArgument,
                                           NewSize,
                                           OldSize);
        Value *Shifted = Extended;
        if (ShiftAmount != 0)
          Shifted = B.CreateLShr(Extended, ShiftAmount);

        Accumulator = B.CreateOr(Accumulator, Shifted);

        OffsetInNewArgument += OldSize;
      }

      if (ModelArgument.Stack and not MaybeStackOffsetAtCallSite) {
        if (not MessageEmitted) {
          MessageEmitted = true;
          emitMessage(OldCall,
                      "Ignoring stack arguments for this call site: "
                      "stack size at call site unknown",
                      OldCall->getDebugLoc());
        }
      } else if (ModelArgument.Stack) {
        unsigned OldSize = ModelArgument.Stack->Size;
        revng_assert(OldSize <= 128 / 8);
        revng_assert(MaybeStackOffsetAtCallSite);

        // Create an alloca
        IntegerType *StackSpanType = B.getIntNTy(OldSize * 8);
        auto Pair = SSA.VariableBuilder.createAllocaWithPtrToInt(Caller,
                                                                 StackSpanType);
        auto [Alloca, PtrToInt] = Pair;

        // Record its portion of the stack for redirection
        RecordStackArgument(*ModelArgument.Stack, PtrToInt);

        Value *Loaded = B.CreateLoad(Alloca->getAllocatedType(), Alloca);

        // Extend, shift and or in Accumulator
        // Note: here we might truncate too, since certain architectures
        //       report a stack span of 8 bytes but the associated type is
        //       actually 32 bits
        Value *Extended = B.CreateZExtOrTrunc(Loaded, LLVMType);

        unsigned ShiftAmount = shiftAmount(OffsetInNewArgument,
                                           NewSize,
                                           OldSize);
        Value *Shifted = Extended;
        if (ShiftAmount != 0)
          Shifted = B.CreateShl(Extended, ShiftAmount);

        Accumulator = B.CreateOr(Accumulator, Shifted);
      }

      Arguments.push_back(Accumulator);
    } break;

    case ArgumentKind::ReferenceToAggregate: {

      SSA.VariableBuilder.setTargetFunction(SSACSCall->getFunction());
      Instruction
        *StackArgsAddress = SSA.VariableBuilder
                              .createCallStackArgumentVariable(*ModelArgument
                                                                  .Type);
      revng_assert(StackArgsAddress);
      // The instruction returning the address of the stack arguments is also
      // the instruction performing the actual allocation, so we can just say
      // they're equal.
      //
      // Also, there's no need to push it ALAP, since it's an alloca.
      // We do have to push ALAP its cast to an integer though.
      revng_assert(isa<PtrToIntInst>(StackArgsAddress));
      revng_assert(isa<AllocaInst>(StackArgsAddress->getOperand(0)));
      Instruction *StackArgsAllocation = StackArgsAddress;
      SSA.ToPushALAP.push_back(StackArgsAddress);

      // We also have to copy over metadata, from the annotation about the
      // size of the stack arguments.
      StackArgsAllocation->copyMetadata(*SSACSCall);

      unsigned OffsetInNewArgument = 0;
      for (auto &Register : ModelArgument.Registers) {
        Value *OldArgument = ArgumentToRegister.at(Register);
        unsigned OldSize = model::Register::getSize(Register);

        Value *Address = createAdd(B, StackArgsAddress, OffsetInNewArgument);

        // Store value
        Value *Pointer = pointer(B, Address);
        B.CreateStore(OldArgument, Pointer);

        OffsetInNewArgument += OldSize;
      }

      if (ModelArgument.Stack) {
        if (MaybeStackOffsetAtCallSite) {
          RecordStackArgument(*ModelArgument.Stack, StackArgsAddress);
        } else {
          if (not MessageEmitted) {
            MessageEmitted = true;
            emitMessage(OldCall,
                        "Ignoring stack arguments for this call site: "
                        "stack size at call site unknown",
                        OldCall->getDebugLoc());
          }
        }
      }

      Arguments.push_back(StackArgsAllocation);
    } break;

    default:
      revng_abort();
    }
  }

  if (Log.isEnabled()) {
    Log << "Redirector data:\n";
    LoggerIndent X(Log);
    Redirector.dump(Log);
    Log << DoLog;
  }

  revng_assert(Redirector.verify());

  // Remove the SPTAR from the argument list, it's not there in the new
  // prototype
  if (HasSPTAR) {
    revng_assert(Arguments.size() > 0);
    Arguments.erase(Arguments.begin());
  }

  // If the old return type and the new one are identical, switch to the old
  // one in the new call
  auto *OldCallType = OldCall->getFunctionType();
  auto *OldReturnType = OldCallType->getReturnType();
  auto *NewReturnType = CalleeType->getReturnType();
  if (auto *OldStructType = dyn_cast<StructType>(OldReturnType)) {
    if (auto *NewStructType = dyn_cast<StructType>(NewReturnType)) {
      if (NewStructType->isLayoutIdentical(OldStructType)) {
        CalleeType = FunctionType::get(OldReturnType,
                                       CalleeType->params(),
                                       CalleeType->isVarArg());
      }
    }
  }

  // Actually create the new call and replace the old one
  CallInst *NewCall = B.CreateCall(CalleeType, CalledValue, Arguments);
  NewCall->copyMetadata(*OldCall);
  NewCall->setAttributes(OldCall->getAttributes());
  const auto &[PointerReturns, PointerArguments] = getPointerMetadata(Layout);
  setPointersMetadata(NewCall, PointerReturns, PointerArguments);

  Value *ReturnValuePointer = nullptr;
  switch (Layout.returnMethod()) {
  case ReturnMethod::ModelAggregate: {
    revng_log(Log, "This call site returns a model aggregate");
    revng_assert(not ReturnValuePointer);
    const auto &ReturnType = Layout.returnValueAggregateType();

    revng_log(Log, "Creating local variable to store the return value");
    auto &VB = SSA.VariableBuilder;
    VB.setTargetFunction(Caller);
    Value *Allocation = nullptr;
    Value *IntAddress = nullptr;
    tie(Allocation,
        IntAddress) = VB.createLocalVariableAndTakeIntAddress(ReturnType);
    B.CreateStore(NewCall, Allocation);
    ReturnValuePointer = IntAddress;

    if (HasSPTAR) {
      if (CallSite.StackReturnValueRange.has_value()) {
        // StackReturnValueRange is already relative to the initial value
        // of the stack pointer
        Redirector.recordSpan(*CallSite.StackReturnValueRange, IntAddress);
      } else {
        revng_log(Log,
                  "Warning: couldn't resolve the location of the pointer "
                  "to the storage for the return value in the SPTAR");
      }
    }

    // We're returning an aggregate, but not via SPTAR, we're using one or
    // more registers
    if (OldReturnType->isStructTy()) {
      SmallVector<SmallPtrSet<CallInst *, 2>, 2>
        ExtractedValues = getExtractedValuesFromInstruction(OldCall);
      for (auto &Group : llvm::enumerate(ExtractedValues)) {

        unsigned FieldIndex = Group.index();
        SmallPtrSet<CallInst *, 2> &ExtractedAtIndex = Group.value();
        if (ExtractedAtIndex.empty())
          continue;

        unsigned BitOffset = getBitOffsetAt(cast<StructType>(OldReturnType),
                                            FieldIndex);
        revng_assert(0 == (BitOffset % 8));
        unsigned ByteOffset = BitOffset / 8;

        Value *Pointer = createAdd(B, ReturnValuePointer, ByteOffset);
        Type *ExtractedType = (*ExtractedAtIndex.begin())->getType();
        auto *Load = B.CreateLoad(ExtractedType, pointer(B, Pointer));

        for (CallInst *Extractor : Group.value()) {
          Extractor->replaceAllUsesWith(Load);
          eraseFromParent(Extractor);
        }
      }

      revng_assert(OldCall->use_empty());
    } else {
      OldCall->replaceAllUsesWith(ReturnValuePointer);
    }

  } break;

  case ReturnMethod::Scalar:

    if (OldReturnType != NewReturnType and OldReturnType->isIntegerTy()
        and NewReturnType->isIntegerTy()) {
      // We're using a large register to return a smaller integer value (e.g.,
      // returning a 32-bit integer through rax, which is 64-bit)
      auto OldSize = OldReturnType->getIntegerBitWidth();
      auto NewSize = NewReturnType->getIntegerBitWidth();
      revng_assert(NewSize <= OldSize);
      auto *Extended = cast<Instruction>(B.CreateZExt(NewCall, OldReturnType));
      OldCall->replaceAllUsesWith(Extended);
    } else if (OldReturnType->isStructTy() and NewReturnType->isIntegerTy()) {
      // We're returning a large integer value through multiple values (e.g.,
      // returning a 64-bit integer through two registers in i386)
      SmallVector<SmallPtrSet<CallInst *, 2>, 2>
        ExtractedValues = getExtractedValuesFromInstruction(OldCall);
      for (auto &Group : llvm::enumerate(ExtractedValues)) {

        unsigned FieldIndex = Group.index();
        SmallPtrSet<CallInst *, 2> &ExtractedAtIndex = Group.value();
        if (ExtractedAtIndex.empty())
          continue;

        unsigned ShiftAmount = getBitOffsetAt(cast<StructType>(OldReturnType),
                                              FieldIndex);
        Type *TruncatedType = (*ExtractedAtIndex.begin())->getType();
        Value *Replacement = B.CreateTrunc(B.CreateLShr(NewCall, ShiftAmount),
                                           TruncatedType);
        for (CallInst *Extractor : Group.value()) {
          revng_assert(TruncatedType == Extractor->getType());
          Extractor->replaceAllUsesWith(Replacement);
          eraseFromParent(Extractor);
        }
      }

      revng_assert(OldCall->use_empty());
    } else {
      revng_assert(not OldReturnType->isStructTy());
      OldCall->replaceAllUsesWith(NewCall);
    }
    break;

  case ReturnMethod::Void:
    // Nothing to do here
    break;
  case ReturnMethod::RegisterSet:
    OldCall->replaceAllUsesWith(NewCall);
    break;

  default:
    revng_abort();
  }

  eraseFromParent(OldCall);
  revng_assert(CalleeType->getPointerTo() == CalledValue->getType());

  return Redirector;
}

bool SegregateStackAccesses::runOnFunction(const model::Function &ModelFunction,
                                           llvm::Function &Function) {
  SegregateFunctionStack Worker(*this, ModelFunction, Function);
  Worker.upgrade();
  Worker.segregate();
  return true;
}

namespace revng::pypeline::piperuns {

// TODO: merge ::SegregateStackAccesses into SegregateStackAccesses once we
//       dismiss the old pipeline
void SegregateStackAccesses::runOnLLVMFunction(const model::Function &Function,
                                               llvm::Function &LLVMFunction) {
  ::SegregateStackAccesses Impl(Binary, *LLVMFunction.getParent());
  Impl.prologue();
  Impl.runOnFunction(Function, LLVMFunction);
  Impl.epilogue();
}

} // namespace revng::pypeline::piperuns
