#pragma once

#include <map>

#include "llvm/IR/GlobalVariable.h"

#include "revng/BasicAnalyses/GeneratedCodeBasicInfo.h"
#include "revng/EarlyFunctionAnalysis/CallGraph.h"
#include "revng/EarlyFunctionAnalysis/Outliner.h"
#include "revng/EarlyFunctionAnalysis/CFGAnalyzer.h"
#include "revng/MFP/MFP.h"
#include "revng/MFP/SetLattices.h"
#include "revng/Model/Register.h"

#include "ABIAnalyses/Analyses.h"

namespace efa {

struct FunctionABI {
  using SetOfRegisters = std::set<model::Register::Values>;
  using SUL = SetUnionLattice<SetOfRegisters>;

  SetOfRegisters ArgumentRegisters;
  SetOfRegisters ReturnRegisters;

  [[nodiscard]] FunctionABI combineValues(const FunctionABI &A2) const {
    FunctionABI Result = {
      .ArgumentRegisters = SUL::combineValues(ArgumentRegisters,
                                              A2.ArgumentRegisters),
      .ReturnRegisters = SUL::combineValues(ReturnRegisters, A2.ReturnRegisters)
    };

    return Result;
  }

  [[nodiscard]] bool isLessOrEqual(const FunctionABI &A2) const {
    if (!SUL::isLessOrEqual(ArgumentRegisters, A2.ArgumentRegisters)) {
      return false;
    }

    return SUL::isLessOrEqual(ReturnRegisters, A2.ReturnRegisters);
  }
};

class LatticeElement {
private:
  using map_t = std::map<MetaAddress, FunctionABI>;
  using iterator_t = map_t::iterator;
  using const_iterator_t = map_t::const_iterator;

  map_t Map;

public:
  const_iterator_t begin() const { return Map.begin(); }
  iterator_t begin() { return Map.begin(); }

  const_iterator_t end() const { return Map.end(); }
  iterator_t end() { return Map.end(); }

  iterator_t find(const map_t::key_type &Key) { return Map.find(Key); }
  const_iterator_t find(const map_t::key_type &Key) const {
    return Map.find(Key);
  }

  map_t::mapped_type &operator[](const auto &Key) {
    return Map[Key];
  }

  map_t::mapped_type &at(const map_t::key_type &Key) {
    return Map.at(Key);
  }

  const map_t::mapped_type &at(const map_t::key_type &Key) const {
    return Map.at(Key);
  }
};

class DetectABIMFI {
private:
  GeneratedCodeBasicInfo &GCBI;
  std::map<MetaAddress, OutlinedFunction> &OutlinedFunctions;
  CFGAnalyzer &Analyzer;
  TupleTree<model::Binary> &Binary;

public:
  using Label = BasicBlockNode *;

  explicit DetectABIMFI(
    GeneratedCodeBasicInfo &GCBI,
    std::map<MetaAddress, OutlinedFunction> &OutlinedFunctions,
    CFGAnalyzer &Analyzer,
    TupleTree<model::Binary> &Binary) :
    GCBI{ GCBI },
    OutlinedFunctions{ OutlinedFunctions },
    Analyzer{ Analyzer },
    Binary{ Binary } {}

  [[nodiscard]] LatticeElement
  combineValues(const LatticeElement &E1, const LatticeElement &E2) const
  {
    LatticeElement Result;

    for (const auto &[Address, E1Abi] : E1) {
      auto E2It = E2.find(Address);
      if (E2It != E2.end()) {
        Result[Address] = E1Abi.combineValues(E2It->second);
      } else {
        Result[Address] = E1Abi;
      }
    }

    for (const auto &[Address, E2Abi] : E2) {
      auto ResultIt = Result.find(Address);
      if (ResultIt == Result.end()) {
        Result[Address] = E2Abi;
      }
    }

    return Result;
  }

  [[nodiscard]] bool
  isLessOrEqual(const LatticeElement &E1, const LatticeElement &E2) const {
    for (const auto &[Address, E1Abi] : E1) {
      auto It = E2.find(Address);
      if (It != E2.end()) {
        if (!E1Abi.isLessOrEqual(It->second)) {
          return false;
        }
      }
    }

    return true;
  }

  [[nodiscard]] LatticeElement
  applyTransferFunction(Label L, const LatticeElement &E2) const
  {
    LatticeElement Result = E2;

    auto Reg = [this](const llvm::GlobalVariable *V) {
      return model::Register::fromCSVName(V->getName(), Binary->Architecture());
    };

    const auto &Address = L->Address;
    if (Address.isInvalid()) {
      return Result;
    }

    auto Entry = GCBI.getBlockAt(Address);
    if (!Entry) {
      return Result;
    }

    using namespace ABIAnalyses;

    ABIAnalysesResults ABIResults;

    OutlinedFunction &OutlinedFunction = getOutlinedFunction(Address);
    auto F = OutlinedFunction.Function.get();

    // Find registers that may be target of at least one store. This helps
    // refine the final results.
    auto WrittenRegisters = findWrittenRegisters(F);

    // Run ABI-independent data-flow analyses
    ABIResults = analyzeOutlinedFunction(F,
                                         GCBI,
                                         Analyzer.preCallHook(),
                                         Analyzer.postCallHook(),
                                         Analyzer.entryHook(),
                                         Analyzer.retHook());

    for (auto &[CSV, State] : ABIResults.ArgumentsRegisters) {
      if (State == abi::RegisterState::Yes) {
        Result[Address].ArgumentRegisters.insert(Reg(CSV));
      }
    }

    for (auto &[CSV, State] : ABIResults.FinalReturnValuesRegisters) {
    }
  }

  static std::set<llvm::GlobalVariable *>
  findWrittenRegisters(llvm::Function *F) {
    using namespace llvm;

    std::set<GlobalVariable *> WrittenRegisters;
    for (auto &BB : *F) {
      for (auto &I : BB) {
        if (auto *SI = dyn_cast<StoreInst>(&I)) {
          Value *Ptr = skipCasts(SI->getPointerOperand());
          if (auto *GV = dyn_cast<GlobalVariable>(Ptr))
            WrittenRegisters.insert(GV);
        }
      }
    }

    return WrittenRegisters;
  }

  OutlinedFunction &getOutlinedFunction(MetaAddress Address) const {
    auto It = OutlinedFunctions.find(Address);
    if (It != OutlinedFunctions.end()) {
      return It->second;
    }

    return OutlinedFunctions[Address];
  }
};

}
