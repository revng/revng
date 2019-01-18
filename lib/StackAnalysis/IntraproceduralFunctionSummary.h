#ifndef INTRAPROCEDURALFUNCTIONSUMMARY_H
#define INTRAPROCEDURALFUNCTIONSUMMARY_H

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// Standard includes
#include <limits>

// Local includes
#include "Element.h"
#include "FunctionABI.h"

extern Logger<> SaLog;

namespace StackAnalysis {

namespace LocalSlotType {

enum Values {
  UsedRegister,
  ExplicitlyCalleeSavedRegister,
  ForwardedArgument,
  ForwardedReturnValue
};

inline const char *getName(Values Type) {
  switch (Type) {
  case UsedRegister:
    return "UsedRegister";
  case ExplicitlyCalleeSavedRegister:
    return "ExplicitlyCalleeSavedRegister";
  case ForwardedArgument:
    return "ForwardedArgument";
  case ForwardedReturnValue:
    return "ForwardedReturnValue";
  }

  revng_abort();
}

} // namespace LocalSlotType

class IntraproceduralFunctionSummary {
public:
  using LocalSlot = std::pair<ASSlot, LocalSlotType::Values>;
  using LocalSlotVector = std::vector<LocalSlot>;
  using IFS = IntraproceduralFunctionSummary;
  using CallSiteStackSizeMap = std::map<FunctionCall, llvm::Optional<int32_t>>;
  using BranchesTypeMap = std::map<llvm::BasicBlock *, BranchType::Values>;

public:
  Intraprocedural::Element FinalState;
  FunctionABI ABI;
  LocalSlotVector LocalSlots;
  CallSiteStackSizeMap FrameSizeAtCallSite;
  BranchesTypeMap BranchesType;
  std::set<int32_t> WrittenRegisters;

private:
  IntraproceduralFunctionSummary() :
    FinalState(Intraprocedural::Element::bottom()) {}

public:
  explicit IntraproceduralFunctionSummary(Intraprocedural::Element FinalState,
                                          FunctionABI ABI,
                                          CallSiteStackSizeMap FrameSizes,
                                          BranchesTypeMap BranchesType,
                                          std::set<int32_t> WrittenRegisters) :
    FinalState(std::move(FinalState)),
    ABI(std::move(ABI)),
    FrameSizeAtCallSite(std::move(FrameSizes)),
    BranchesType(std::move(BranchesType)),
    WrittenRegisters(std::move(WrittenRegisters)) {

    process();
  }

  static IntraproceduralFunctionSummary bottom() {
    return IntraproceduralFunctionSummary();
  }

  IFS copy() const {
    IFS Result;
    Result.FinalState = FinalState.copy();
    Result.ABI = ABI.copy();
    Result.LocalSlots = LocalSlots;
    Result.FrameSizeAtCallSite = FrameSizeAtCallSite;
    Result.BranchesType = BranchesType;
    Result.WrittenRegisters = WrittenRegisters;
    return Result;
  }

  IntraproceduralFunctionSummary(const IFS &) = delete;
  IntraproceduralFunctionSummary &operator=(const IFS &) = delete;

  IntraproceduralFunctionSummary(IFS &&) = default;
  IntraproceduralFunctionSummary &operator=(IFS &&) = default;

  void dump(const llvm::Module *M) const debug_function { dump(M, dbg); }

  template<typename T>
  void dump(const llvm::Module *M, T &Output) const {
    Output << "FinalState:\n";
    FinalState.dump(M, Output);
    Output << "\n";

    Output << "ABI:\n";
    ABI.dump(M, Output);
    Output << "\n";

    Output << "Local slots (" << LocalSlots.size() << "):\n";
    for (const LocalSlot &Slot : LocalSlots) {
      Output << "  ";
      Slot.first.dump(M, Output);
      Output << ": " << LocalSlotType::getName(Slot.second) << "\n";
    }
  }

private:
  void process() {
    using namespace Intraprocedural;
    using std::set;

    auto CPU = ASID::cpuID();
    auto SP0 = ASID::stackID();

    const llvm::Module *M = nullptr;
    int32_t CSVCount = std::numeric_limits<int32_t>::max();
    if (BranchesType.size() > 0) {
      M = getModule(BranchesType.begin()->first);
      CSVCount = std::distance(M->global_begin(), M->global_end());
    }

    auto IsValid = [CSVCount, CPU](ASSlot Slot) {
      return Slot.addressSpace() == CPU and Slot.offset() <= CSVCount;
    };

    // Collect slots in the summary and those obtained by computing the ECS
    // slots
    set<ASSlot> SlotsPool = FinalState.collectSlots(CSVCount);
    ABI.collectLocalSlots(SlotsPool);
    set<ASSlot> CalleeSaved = FinalState.computeCalleeSavedSlots();

    revng_assert(std::all_of(SlotsPool.begin(), SlotsPool.end(), IsValid));

    for (ASSlot Slot : CalleeSaved)
      if (Slot.addressSpace() == ASID::cpuID())
        SlotsPool.insert(Slot);
    revng_assert(std::all_of(SlotsPool.begin(), SlotsPool.end(), IsValid));

    set<ASSlot> ForwardedArguments;
    set<ASSlot> ForwardedReturnValues;

    set<int32_t> Arguments;
    set<int32_t> ReturnValues;
    std::tie(Arguments, ReturnValues) = ABI.collectYesRegisters();

    // Loop over return values to identify forwarded arguments (push rax; pop
    // rdx)
    for (int32_t Register : ReturnValues) {
      ASSlot RegisterSlot = ASSlot::create(CPU, Register);
      Value Content = FinalState.load(Value::fromSlot(RegisterSlot));
      if (const ASSlot *TheTag = Content.tag()) {
        if (TheTag->addressSpace() == CPU and Register != TheTag->offset()
            and Arguments.count(TheTag->offset()) != 0) {
          // We have a return value containing the initial value of (another)
          // argument

          // Check if we have this value in a stack slot too
          if (FinalState.addressSpaceContainsTag(SP0, TheTag)) {
            // OK, this is a forwarded argument
            ForwardedArguments.insert(*TheTag);
            ForwardedReturnValues.insert(ASSlot::create(CPU, Register));
          }
        }
      }
    }

    // Sort out CPU slots by type
    for (ASSlot Slot : SlotsPool) {
      revng_assert(Slot.addressSpace() == CPU);
      if (CalleeSaved.count(Slot) != 0) {
        LocalSlots.emplace_back(Slot,
                                LocalSlotType::ExplicitlyCalleeSavedRegister);
      } else if (ForwardedArguments.count(Slot) != 0) {
        LocalSlots.emplace_back(Slot, LocalSlotType::ForwardedArgument);
      } else if (ForwardedReturnValues.count(Slot) != 0) {
        LocalSlots.emplace_back(Slot, LocalSlotType::ForwardedReturnValue);
      } else {
        LocalSlots.emplace_back(Slot, LocalSlotType::UsedRegister);
      }
    }

    for (const LocalSlot &Slot : LocalSlots) {
      switch (Slot.second) {
      case LocalSlotType::ExplicitlyCalleeSavedRegister:
        // Drop from ABI analyses, pretend nothing happened
        ABI.drop(Slot.first);
        break;

      case LocalSlotType::ForwardedArgument:
      case LocalSlotType::ForwardedReturnValue:
        ABI.resetToUnknown(Slot.first);
        break;

      case LocalSlotType::UsedRegister:
        break;
      }
    }
  }
};

} // namespace StackAnalysis

#endif // INTRAPROCEDURALFUNCTIONSUMMARY_H
