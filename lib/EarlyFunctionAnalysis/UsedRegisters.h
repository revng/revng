#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <map>
#include <set>

#include "revng/EarlyFunctionAnalysis/CallGraph.h"
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

// using LatticeElement = std::map<MetaAddress, FunctionABI>;

struct LatticeElement : std::map<MetaAddress, FunctionABI> {
  virtual ~LatticeElement() = default;

  [[nodiscard]] virtual bool
  hasRetReg(MetaAddress Addr, model::Register::Values V) const {
    if (count(Addr)) {
      return at(Addr).ReturnRegisters.count(V);
    }

    return false;
  }

  [[nodiscard]] virtual bool
  hasArgReg(MetaAddress Addr, model::Register::Values V) const {
    if (count(Addr)) {
      return at(Addr).ArgumentRegisters.count(V);
    }

    return false;
  }
};

struct ExtremalLatticeElement : LatticeElement {
  [[nodiscard]] bool
  hasRetReg(MetaAddress Addr, model::Register::Values V) const {
    return true;
  }

  [[nodiscard]] bool
  hasArgReg(MetaAddress Addr, model::Register::Values V) const {
    return true;
  }
};

struct UsedRegistersMFI {
  using Label = BasicBlockNode *;
  using GraphType = llvm::Inverse<BasicBlockNode *>;
  using LatticeElement = LatticeElement;
  using ExtremalLatticeElement = ExtremalLatticeElement;

  DetectABI &DA;

  explicit UsedRegistersMFI(DetectABI &DA) : DA{ DA } {}

  [[nodiscard]] LatticeElement
  combineValues(const LatticeElement &E1, const LatticeElement &E2) const {
    LatticeElement Out;
    for (const auto &[Address, E1Abi] : E1) {
      if (E2.count(Address)) {
        const auto &E2Abi = E2.at(Address);
        const auto &E1Args = E1Abi.ArgumentRegisters;
        const auto &E1Ret = E1Abi.ReturnRegisters;
        const auto &E2Args = E2Abi.ArgumentRegisters;
        const auto &E2Ret = E2Abi.ReturnRegisters;
        Out[Address]
          .ArgumentRegisters = FunctionABI::SUL::combineValues(E1Args, E2Args);
        Out[Address].ReturnRegisters = FunctionABI::SUL::combineValues(E1Ret,
                                                                       E2Ret);
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
  isLessOrEqual(const LatticeElement &Left, const LatticeElement &Right) const {
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
  applyTransferFunction(Label L, const LatticeElement &E2) const {
    LatticeElement Result = E2;
    auto Reg = [this](const llvm::GlobalVariable *V) {
      return model::Register::fromCSVName(V->getName(),
                                          DA.getBinary()->Architecture());
    };
    for (const auto &[Address, ABI] : E2) {
      auto Results = DA.analyzeABI(Address);
      for (auto &[CSV, State] : Results.ArgumentsRegisters) {
        Result[Address].ArgumentRegisters.insert(Reg(CSV));
      }

      for (auto &[CSV, State] : Results.FinalReturnValuesRegisters) {
        Result[Address].ReturnRegisters.insert(Reg(CSV));
      }
    }

    return Result;
  }

  static ExtremalLatticeElement ExtremalLattice;
};

} // namespace efa
