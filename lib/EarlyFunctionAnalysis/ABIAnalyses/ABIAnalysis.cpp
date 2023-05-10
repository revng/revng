//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"

#include "revng/ABI/RegisterState.h"
#include "revng/ADT/ZipMapIterator.h"
#include "revng/EarlyFunctionAnalysis/ABIAnalysis.h"
#include "revng/EarlyFunctionAnalysis/Common.h"
#include "revng/Model/Binary.h"
#include "revng/Support/Assert.h"
#include "revng/Support/Debug.h"
#include "revng/Support/IRHelpers.h"
#include "revng/Support/MetaAddress.h"

#include "Analyses.h"

using namespace llvm;

static Logger<> ABIAnalysesLog("abi-analyses");

namespace ABIAnalyses {
using RegisterState = abi::RegisterState::Values;

template void
ABIAnalyses::ABIAnalysesResults::dump<Logger<true>>(Logger<true> &,
                                                    const char *) const;

struct PartialAnalysisResults {
  // Per function analysis
  RegisterStateMap UAOF;
  RegisterStateMap DRAOF;

  // Per call site analysis
  std::map<std::pair<BasicBlockID, BasicBlock *>, RegisterStateMap> URVOFC;
  std::map<std::pair<BasicBlockID, BasicBlock *>, RegisterStateMap> RAOFC;
  std::map<std::pair<BasicBlockID, BasicBlock *>, RegisterStateMap> DRVOFC;

  // Per return analysis
  std::map<std::pair<BasicBlockID, BasicBlock *>, RegisterStateMap> URVOF;

  // Debug methods
  void dump() const debug_function { dump(dbg, ""); }

  template<typename T>
  void dump(T &Output, const char *Prefix) const;
};

// Print the analysis results
template<typename T>
void PartialAnalysisResults::dump(T &Output, const char *Prefix) const {
  Output << Prefix << "UsedArgumentsOfFunction:\n";
  for (auto &[GV, State] : UAOF) {
    Output << Prefix << "  " << GV->getName().str() << " = "
           << abi::RegisterState::getName(State).str() << '\n';
  }

  Output << Prefix << "UsedReturnValuesOfFunctionCall:\n";
  for (auto &[Key, StateMap] : URVOFC) {
    Output << Prefix << "  " << Key.second->getName().str() << '\n';
    for (auto &[GV, State] : StateMap) {
      Output << Prefix << "    " << GV->getName().str() << " = "
             << abi::RegisterState::getName(State).str() << '\n';
    }
  }

  Output << Prefix << "RegisterArgumentsOfFunctionCall:\n";
  for (auto &[Key, StateMap] : RAOFC) {
    Output << Prefix << "  " << Key.second->getName().str() << '\n';
    for (auto &[GV, State] : StateMap) {
      Output << Prefix << "    " << GV->getName().str() << " = "
             << abi::RegisterState::getName(State).str() << '\n';
    }
  }

  Output << Prefix << "UsedReturnValuesOfFunction:\n";
  for (auto &[Key, StateMap] : URVOF) {
    Output << Prefix << "  " << Key.second->getName().str() << '\n';
    for (auto &[GV, State] : StateMap) {
      Output << Prefix << "  " << GV->getName().str() << " = "
             << abi::RegisterState::getName(State).str() << '\n';
    }
  }
}

RegisterState combine(RegisterState LH, RegisterState RH) {
  switch (LH) {
  case RegisterState::Yes:
    switch (RH) {
    case RegisterState::Yes:
    case RegisterState::YesOrDead:
    case RegisterState::Maybe:
      return RegisterState::Yes;
    case RegisterState::No:
    case RegisterState::NoOrDead:
    case RegisterState::Dead:
    case RegisterState::Contradiction:
      return RegisterState::Contradiction;
    case RegisterState::Count:
    case RegisterState::Invalid:
      revng_abort();
    }
    break;

  case RegisterState::YesOrDead:
    switch (RH) {
    case RegisterState::Yes:
      return RegisterState::Yes;
    case RegisterState::Maybe:
    case RegisterState::YesOrDead:
      return RegisterState::YesOrDead;
    case RegisterState::Dead:
    case RegisterState::NoOrDead:
      return RegisterState::Dead;
    case RegisterState::No:
    case RegisterState::Contradiction:
      return RegisterState::Contradiction;
    case RegisterState::Count:
    case RegisterState::Invalid:
      revng_abort();
    }
    break;

  case RegisterState::No:
    switch (RH) {
    case RegisterState::No:
    case RegisterState::NoOrDead:
    case RegisterState::Maybe:
      return RegisterState::No;
    case RegisterState::Yes:
    case RegisterState::YesOrDead:
    case RegisterState::Dead:
    case RegisterState::Contradiction:
      return RegisterState::Contradiction;
    case RegisterState::Count:
    case RegisterState::Invalid:
      revng_abort();
    }
    break;

  case RegisterState::NoOrDead:
    switch (RH) {
    case RegisterState::No:
      return RegisterState::No;
    case RegisterState::Maybe:
    case RegisterState::NoOrDead:
      return RegisterState::NoOrDead;
    case RegisterState::Dead:
    case RegisterState::YesOrDead:
      return RegisterState::Dead;
    case RegisterState::Yes:
    case RegisterState::Contradiction:
      return RegisterState::Contradiction;
    case RegisterState::Count:
    case RegisterState::Invalid:
      revng_abort();
    }
    break;

  case RegisterState::Dead:
    switch (RH) {
    case RegisterState::Dead:
    case RegisterState::Maybe:
    case RegisterState::NoOrDead:
    case RegisterState::YesOrDead:
      return RegisterState::Dead;
    case RegisterState::No:
    case RegisterState::Yes:
    case RegisterState::Contradiction:
      return RegisterState::Contradiction;
    case RegisterState::Count:
    case RegisterState::Invalid:
      revng_abort();
    }
    break;

  case RegisterState::Maybe:
    return RH;

  case RegisterState::Contradiction:
    return RegisterState::Contradiction;

  case RegisterState::Count:
  case RegisterState::Invalid:
    revng_abort();
  }
}

void finalizeReturnValues(ABIAnalysesResults &ABIResults) {
  for (auto &[PC, RSMap] : ABIResults.ReturnValuesRegisters) {
    for (auto &[CSV, RS] : RSMap) {
      if (ABIResults.FinalReturnValuesRegisters.count(CSV) == 0)
        ABIResults.FinalReturnValuesRegisters[CSV] = RegisterState::Maybe;

      ABIResults.FinalReturnValuesRegisters
        [CSV] = combine(ABIResults.FinalReturnValuesRegisters[CSV], RS);
    }
  }
}

template<typename T>
void ABIAnalysesResults::dump(T &Output, const char *Prefix) const {
  Output << Prefix << "Arguments:\n";
  for (auto &[GV, State] : ArgumentsRegisters) {
    Output << Prefix << "  " << GV->getName().str() << " = "
           << abi::RegisterState::getName(State).str() << '\n';
  }

  Output << Prefix << "Call site:\n";
  for (auto &[PC, StateMap] : CallSites) {
    Output << Prefix << "  " << PC.toString() << '\n';
    Output << Prefix << "  "
           << "  "
           << "Arguments:\n";
    for (auto &[GV, State] : StateMap.ArgumentsRegisters) {
      Output << Prefix << "      " << GV->getName().str() << " = "
             << abi::RegisterState::getName(State).str() << '\n';
    }
    Output << Prefix << "  "
           << "  "
           << "Return values:\n";
    for (auto &[GV, State] : StateMap.ReturnValuesRegisters) {
      Output << Prefix << "      " << GV->getName().str() << " = "
             << abi::RegisterState::getName(State).str() << '\n';
    }
  }

  Output << Prefix << "Return values:\n";
  for (auto &[PC, StateMap] : ReturnValuesRegisters) {
    for (auto &[GV, State] : StateMap) {
      Output << Prefix << "  " << GV->getName().str() << " = "
             << abi::RegisterState::getName(State).str() << '\n';
    }
  }

  Output << Prefix << "Final Return values:\n";
  for (auto &[GV, State] : FinalReturnValuesRegisters) {
    Output << Prefix << "  " << GV->getName().str() << " = "
           << abi::RegisterState::getName(State).str() << '\n';
  }
}

} // namespace ABIAnalyses
