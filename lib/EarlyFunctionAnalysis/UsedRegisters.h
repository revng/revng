#pragma once

#include <map>
#include <set>

#include "revng/EarlyFunctionAnalysis/Common.h"
#include "revng/MFP/SetLattices.h"
#include "revng/Model/Register.h"
#include "revng/Support/MetaAddress.h"

#include "DetectABI.h"

namespace efa {

struct FunctionABI {
  using SetOfRegisters = std::set<model::Register::Values>;
  using SUL = SetUnionLattice<SetOfRegisters>;
  SetOfRegisters ArgumentRegisters;
  SetOfRegisters ReturnRegisters;
};

using LatticeElement = std::map<MetaAddress, FunctionABI>;

struct UsedRegistersMFI {
  using Label = int;

  DetectABI &DA;

  explicit UsedRegistersMFI(DetectABI &DA) : DA{ DA } {}

  [[nodiscard]] LatticeElement
  combine(const LatticeElement &E1, const LatticeElement &E2) {
    LatticeElement Out;
    for (const auto &[Address, E1Abi] : E1) {
      if (E2.count(Address)) {
        const auto &E2Abi = E2.at(Address);
        Out[Address].ArgumentRegisters = FunctionABI::SUL::
          combineValues(E1Abi.ArgumentRegisters, E2Abi.ArgumentRegisters);
        Out[Address].ReturnRegisters = FunctionABI::SUL::
          combineValues(E1Abi.ReturnRegisters, E2Abi.ReturnRegisters);
      } else {
        Out[Address] = E1Abi;
      }
    }

    for (const auto &[Address, E2Abi] : E2) {
      if (!Out.count(Address)) {
        Out[Address] = E2Abi;
      }
    }

    return Out;
  }

  [[nodiscard]] bool
  isLessOrEqual(const LatticeElement &Left, const LatticeElement &Right) {
    for (const auto &[Address, LeftAbi] : Left) {
      if (Right.count(Address)) {
        const auto &RightAbi = Right.at(Address);
        if (!FunctionABI::SUL::isLessOrEqual(LeftAbi.ArgumentRegisters,
                                             RightAbi.ArgumentRegisters)) {
          return false;
        }

        if (!FunctionABI::SUL::isLessOrEqual(LeftAbi.ReturnRegisters,
                                             RightAbi.ReturnRegisters)) {
          return false;
        }
      }
    }

    return true;
  }

  [[nodiscard]] LatticeElement
  applyTransferFunction(Label L,
                        const LatticeElement &E2) { // TODO: Set label type
    for (const auto &[Address, ABI] : E2) {
      analyzeABI(GCBI.getBlockAt(Address));
      // TODO: get registers from result and put into output LatticeElement
    }

    return LatticeElement{};
  }

  void analyzeABI(llvm::BasicBlock *Entry) { (void) (Entry); }
};

}  // namespace efa
