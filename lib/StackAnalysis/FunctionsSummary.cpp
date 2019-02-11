/// \file abianalyses.cpp
/// \brief Implementation of the classes representing an argument/return value
///        in a register

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// Boost includes
#include <boost/icl/interval_set.hpp>

// Local libraries includes
#include "revng/StackAnalysis/FunctionsSummary.h"
#include "revng/Support/IRHelpers.h"

// Local includes
#include "ASSlot.h"

using llvm::BasicBlock;
using llvm::BlockAddress;
using llvm::CallInst;
using llvm::cast;
using llvm::cast_or_null;
using llvm::dyn_cast;
using llvm::dyn_cast_or_null;
using llvm::GlobalVariable;
using llvm::MDNode;
using llvm::MDString;
using llvm::MDTuple;
using llvm::Metadata;
using llvm::Module;
using llvm::SmallVector;
using llvm::StringRef;
using llvm::TerminatorInst;
using llvm::User;

namespace StackAnalysis {

template class RegisterArgument<true>;
template class RegisterArgument<false>;

using FRA = FunctionRegisterArgument;
using FCRA = FunctionCallRegisterArgument;

template<>
void FRA::combine(const FCRA &Other) {
  // TODO: we're handling this as a special case
  if (Value == No || Other.Value == FunctionCallRegisterArgument::No) {
    Value = No;
    return;
  }

  revng_assert(Other.Value == FunctionCallRegisterArgument::Maybe
               || Other.Value == FunctionCallRegisterArgument::Yes);

  revng_assert(Value == NoOrDead || Value == Maybe || Value == Contradiction
               || Value == Yes || Value == Dead);

  if (Other.Value == FunctionCallRegisterArgument::Yes) {
    switch (Value) {
    case NoOrDead:
      Value = Dead;
      break;
    case Maybe:
      Value = Yes;
      break;
    case Contradiction:
      Value = Contradiction;
      break;
    case Yes:
      Value = Yes;
      break;
    case Dead:
      Value = Dead;
      break;
    case No:
      revng_abort();
    }
  } else {
    switch (Value) {
    case NoOrDead:
      Value = NoOrDead;
      break;
    case Maybe:
      Value = Maybe;
      break;
    case Contradiction:
      Value = Contradiction;
      break;
    case Yes:
      Value = Yes;
      break;
    case Dead:
      Value = Dead;
      break;
    case No:
      revng_abort();
    }
  }
}

template<>
void FCRA::combine(const FRA &Other) {
  // TODO: we're handling this as a special case
  if (Value == No || Other.Value == FunctionRegisterArgument::No) {
    Value = No;
    return;
  }

  revng_assert(Value == Maybe || Value == Yes);

  revng_assert(Other.Value == FunctionRegisterArgument::NoOrDead
               || Other.Value == FunctionRegisterArgument::Maybe
               || Other.Value == FunctionRegisterArgument::Contradiction
               || Other.Value == FunctionRegisterArgument::Yes);

  if (Value == Yes) {
    switch (Other.Value) {
    case FunctionRegisterArgument::NoOrDead:
      Value = Dead;
      break;
    case FunctionRegisterArgument::Maybe:
      Value = Yes;
      break;
    case FunctionRegisterArgument::Contradiction:
      Value = Contradiction;
      break;
    case FunctionRegisterArgument::Yes:
      Value = Yes;
      break;
    default:
      revng_abort();
    }
  } else {
    switch (Other.Value) {
    case FunctionRegisterArgument::NoOrDead:
      Value = NoOrDead;
      break;
    case FunctionRegisterArgument::Maybe:
      Value = Maybe;
      break;
    case FunctionRegisterArgument::Contradiction:
      Value = Contradiction;
      break;
    case FunctionRegisterArgument::Yes:
      Value = Yes;
      break;
    default:
      revng_abort();
    }
  }
}

void FunctionReturnValue::combine(const FunctionCallReturnValue &Other) {
  // TODO: we're handling this as a special case
  if (Value == No || Other.Value == FunctionCallReturnValue::No) {
    Value = No;
    return;
  }

  // *this has seen only URVOF, which can only have Maybe or Yes value
  revng_assert(Value == Maybe || Value == YesCandidate || Value == Dead
               || Value == Yes || Value == NoOrDead);

  // Other is affected by URVOFC and DRVOFC, so that possible states are Maybe,
  // NoOrDead and Yes
  revng_assert(Other.Value == FunctionCallReturnValue::Maybe
               || Other.Value == FunctionCallReturnValue::Contradiction
               || Other.Value == FunctionCallReturnValue::NoOrDead
               || Other.Value == FunctionCallReturnValue::Yes);

  switch (Value) {
  case YesCandidate:
    switch (Other.Value) {
    case FunctionCallReturnValue::Maybe:
      Value = YesCandidate;
      break;
    case FunctionCallReturnValue::NoOrDead:
      Value = Dead;
      break;
    case FunctionCallReturnValue::Yes:
      Value = Yes;
      break;
    case FunctionCallReturnValue::Contradiction:
      Value = Contradiction;
      break;
    default:
      revng_abort();
    }
    break;
  case Maybe:
    switch (Other.Value) {
    case FunctionCallReturnValue::Maybe:
      Value = Maybe;
      break;
    case FunctionCallReturnValue::NoOrDead:
      Value = NoOrDead;
      break;
    case FunctionCallReturnValue::Yes:
      Value = Yes;
      break;
    case FunctionCallReturnValue::Contradiction:
      Value = Contradiction;
      break;
    default:
      revng_abort();
    }
    break;
  case Dead:
    switch (Other.Value) {
    case FunctionCallReturnValue::Maybe:
      Value = Dead;
      break;
    case FunctionCallReturnValue::NoOrDead:
      Value = Dead;
      break;
    case FunctionCallReturnValue::Yes:
      Value = Yes;
      break;
    case FunctionCallReturnValue::Contradiction:
      Value = Contradiction;
      break;
    default:
      revng_abort();
    }
    break;
  case Yes:
    switch (Other.Value) {
    case FunctionCallReturnValue::Maybe:
      Value = Yes;
      break;
    case FunctionCallReturnValue::NoOrDead:
      // TODO: contradiction?
      Value = Yes;
      break;
    case FunctionCallReturnValue::Yes:
      Value = Yes;
      break;
    case FunctionCallReturnValue::Contradiction:
      Value = Contradiction;
      break;
    default:
      revng_abort();
    }
    break;
  case NoOrDead:
    switch (Other.Value) {
    case FunctionCallReturnValue::Maybe:
      Value = NoOrDead;
      break;
    case FunctionCallReturnValue::NoOrDead:
      Value = NoOrDead;
      break;
    case FunctionCallReturnValue::Yes:
      Value = Yes;
      break;
    case FunctionCallReturnValue::Contradiction:
      Value = Contradiction;
      break;
    default:
      revng_abort();
    }
    break;
  default:
    revng_abort();
  }
}

void FunctionCallReturnValue::combine(const FunctionReturnValue &Other) {
  // TODO: we're handling this as a special case
  if (Value == No || Other.Value == FunctionReturnValue::No) {
    Value = No;
    return;
  }

  // *this has seen only URVOF, which can only have Maybe or Yes value
  revng_assert(Other.Value == FunctionReturnValue::Maybe
               || Other.Value == FunctionReturnValue::YesCandidate);

  // Other is affected by URVOFC and DRVOFC, so that possible states are Maybe,
  // NoOrDead, Yes and Contradiction
  revng_assert(Value == Maybe || Value == NoOrDead || Value == Yes
               || Value == Contradiction);

  if (Other.Value == FunctionReturnValue::YesCandidate) {
    switch (Value) {
    case Maybe:
      Value = Yes;
      break;
    case NoOrDead:
      Value = Dead;
      break;
    case Yes:
      Value = Yes;
      break;
    case Contradiction:
      Value = Contradiction;
      break;
    default:
      revng_abort();
    }
  } else {
    switch (Value) {
    case Maybe:
      Value = Maybe;
      break;
    case NoOrDead:
      Value = NoOrDead;
      break;
    case Yes:
      Value = Yes;
      break;
    case Contradiction:
      Value = Contradiction;
      break;
    default:
      revng_abort();
    }
  }
}

void FunctionsSummary::dumpInternal(const Module *M,
                                    StreamWrapperBase &&Stream) const {
  std::stringstream Output;

  // Register the range of addresses covered by each basic block
  using interval_set = boost::icl::interval_set<uint64_t>;
  using interval = boost::icl::interval<uint64_t>;
  std::map<BasicBlock *, interval_set> Coverage;
  for (User *U : M->getFunction("newpc")->users()) {
    auto *Call = dyn_cast<CallInst>(U);
    if (Call == nullptr)
      continue;

    BasicBlock *BB = Call->getParent();
    uint64_t Address = getLimitedValue(Call->getOperand(0));
    uint64_t Size = getLimitedValue(Call->getOperand(1));
    revng_assert(Address > 0 && Size > 0);

    Coverage[BB] += interval::right_open(Address, Address + Size);
  }

  // Sort the functions by name, for extra determinism!
  using Pair = std::pair<BasicBlock *, const FunctionDescription *>;
  std::vector<Pair> SortedFunctions;
  for (auto &P : Functions)
    SortedFunctions.push_back({ P.first, &P.second });
  auto Compare = [](const Pair &A, const Pair &B) {
    return getName(A.first) < getName(B.first);
  };
  std::sort(SortedFunctions.begin(), SortedFunctions.end(), Compare);

  const char *FunctionDelimiter = "";
  Output << "[";
  for (auto &P : SortedFunctions) {
    Output << FunctionDelimiter << "\n  {\n";
    BasicBlock *Entry = P.first;
    const FunctionDescription &Function = *P.second;

    Output << "    \"entry_point\": \"";
    if (Entry != nullptr)
      Output << getName(Entry);
    Output << "\",\n";
    Output << "    \"entry_point_address\": \"";
    if (Entry != nullptr)
      Output << "0x" << std::hex << getBasicBlockPC(Entry);
    Output << "\",\n";

    Output << "    \"jt-reasons\": [";
    if (Entry != nullptr) {
      const char *JTReasonsDelimiter = "";
      TerminatorInst *T = Entry->getTerminator();
      revng_assert(T != nullptr);
      MDNode *Node = T->getMetadata("revng.jt.reasons");
      SmallVector<StringRef, 4> Reasons;
      if (auto *Tuple = cast_or_null<MDTuple>(Node)) {
        // Collect reasons
        for (Metadata *ReasonMD : Tuple->operands())
          Reasons.push_back(cast<MDString>(ReasonMD)->getString());

        // Sort the output to make it more deterministic
        std::sort(Reasons.begin(), Reasons.end());

        // Print out
        for (StringRef Reason : Reasons) {
          Output << JTReasonsDelimiter << "\"" << Reason.data() << "\"";
          JTReasonsDelimiter = ", ";
        }
      }
    }
    Output << "],\n";

    Output << "    \"type\": \"" << getName(Function.Type) << "\",\n";

    interval_set FunctionCoverage;

    const char *BasicBlockDelimiter = "";
    Output << "    \"basic_blocks\": [";

    // Sort basic blocks by name
    using Pair = std::pair<BasicBlock *const, BranchType::Values>;
    std::vector<const Pair *> SortedBasicBlocks;
    for (const Pair &P : Function.BasicBlocks)
      SortedBasicBlocks.push_back(&*Function.BasicBlocks.find(P.first));
    auto Compare = [](const Pair *P, const Pair *Q) {
      return P->first->getName() < Q->first->getName();
    };
    std::sort(SortedBasicBlocks.begin(), SortedBasicBlocks.end(), Compare);

    for (const Pair *P : SortedBasicBlocks) {
      BasicBlock *BB = P->first;
      BranchType::Values Type = P->second;
      const char *TypeName = BranchType::getName(Type);
      Output << BasicBlockDelimiter;
      Output << "{\"name\": \"" << getName(BB) << "\", ";
      Output << "\"type\": \"" << TypeName << "\", ";
      auto It = Coverage.find(BB);
      if (It != Coverage.end()) {
        const interval_set &IntervalSet = It->second;
        FunctionCoverage += IntervalSet;
        revng_assert(IntervalSet.iterative_size() == 1);
        const auto &Range = *(IntervalSet.begin());
        Output << "\"start\": \"0x" << std::hex << Range.lower() << "\", ";
        Output << "\"end\": \"0x" << std::hex << Range.upper() << "\"";
      } else {
        Output << "\"start\": \"\", \"end\": \"\"";
      }
      Output << "}";
      BasicBlockDelimiter = ", ";
    }
    Output << "],\n";

    Output << "    \"slots\": [";
    const char *SlotDelimiter = "";
    for (auto &Q : Function.RegisterSlots) {
      Output << SlotDelimiter;
      Output << "{\"slot\": \"" << Q.first->getName().data() << "\", ";

      Output << "\"argument\": \"";
      Q.second.Argument.dump(Output);
      Output << "\", ";
      Output << "\"return_value\": \"";
      Q.second.ReturnValue.dump(Output);
      Output << "\"}";
      SlotDelimiter = ", ";
    }
    Output << "],\n";

    Output << "    \"clobbered\": [";
    const char *ClobberedDelimiter = "";
    for (const GlobalVariable *CSV : Function.ClobberedRegisters) {
      Output << ClobberedDelimiter;
      Output << "\"" << CSV->getName().data() << "\"";
      ClobberedDelimiter = ", ";
    }
    Output << "],\n";

    const char *CoverageDelimiter = "";
    Output << "    \"coverage\": [";
    for (const auto &Range : FunctionCoverage) {
      Output << CoverageDelimiter;
      Output << "{";
      Output << "\"start\": \"0x" << std::hex << Range.lower() << "\", ";
      Output << "\"end\": \"0x" << std::hex << Range.upper() << "\"";
      Output << "}";
      CoverageDelimiter = ", ";
    }
    Output << "],\n";

    const char *FunctionCallDelimiter = "";
    Output << "    \"function_calls\": [";
    for (const CallSiteDescription &CallSite : Function.CallSites) {
      Output << FunctionCallDelimiter << "\n";
      Output << "      {\n";
      Output << "        \"caller\": ";
      Output << "\"" << getName(CallSite.Call) << "\",\n";
      // TODO: caller address
      // TODO: callee
      // TODO: callee address
      Output << "        \"slots\": [";
      const char *FunctionCallSlotsDelimiter = "";
      for (auto &Q : CallSite.RegisterSlots) {
        Output << FunctionCallSlotsDelimiter;
        Output << "{\"slot\": \"" << Q.first->getName().data() << "\", ";

        Output << "\"argument\": \"";
        Q.second.Argument.dump(Output);
        Output << "\", ";
        Output << "\"return_value\": \"";
        Q.second.ReturnValue.dump(Output);
        Output << "\"}";

        FunctionCallSlotsDelimiter = ", ";
      }
      Output << "]\n";

      Output << "      }";
      FunctionCallDelimiter = ",";
    }
    Output << "\n    ]\n";

    Output << "  }";
    FunctionDelimiter = ",";

    Stream.flush(Output);
  }
  Output << "\n]\n";

  Stream.flush(Output);
}

} // namespace StackAnalysis
