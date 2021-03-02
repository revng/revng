#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// This file has been automatically generated from
// scripts/monotone-framework-lattice.py, please don't change it

#include "revng/ABIAnalyses/Common.h"
#include "revng/Model/Binary.h"

namespace DeadReturnValuesOfFunctionCall {

struct CoreLattice {

  using LatticeElement = model::RegisterState::Values;
  using TransferFunction = ABIAnalyses::TransferKind;

  static bool
  isLessOrEqual(const LatticeElement &Lh, const LatticeElement &Rh) {
    return Lh == Rh
           || (Lh == LatticeElement::Maybe && Rh == LatticeElement::Unknown)
           || (Lh == LatticeElement::NoOrDead && Rh == LatticeElement::Maybe)
           || (Lh == LatticeElement::NoOrDead && Rh == LatticeElement::Unknown);
  }

  static LatticeElement
  combineValues(const LatticeElement &Lh, const LatticeElement &Rh) {
    if ((Lh == LatticeElement::Maybe && Rh == LatticeElement::NoOrDead)
        || (Lh == LatticeElement::NoOrDead && Rh == LatticeElement::Maybe)) {
      return LatticeElement::Maybe;
    } else if ((Lh == LatticeElement::Maybe && Rh == LatticeElement::Unknown)
               || (Lh == LatticeElement::NoOrDead
                   && Rh == LatticeElement::Unknown)
               || (Lh == LatticeElement::Unknown && Rh == LatticeElement::Maybe)
               || (Lh == LatticeElement::Unknown
                   && Rh == LatticeElement::NoOrDead)) {
      return LatticeElement::Unknown;
    }
    return Lh;
  }

  static LatticeElement transfer(TransferFunction T, const LatticeElement &E) {
    switch (T) {
    case TransferFunction::Read:
      switch (E) {
      case LatticeElement::Maybe:
        return LatticeElement::Unknown;
      case LatticeElement::Unknown:
        return LatticeElement::Unknown;
      case LatticeElement::NoOrDead:
        return LatticeElement::NoOrDead;
      default:
        return E;
      }
      return E;

    case TransferFunction::TheCall:
      switch (E) {
      case LatticeElement::Maybe:
        return LatticeElement::Unknown;
      case LatticeElement::Unknown:
        return LatticeElement::Unknown;
      case LatticeElement::NoOrDead:
        return LatticeElement::NoOrDead;
      default:
        return E;
      }
      return E;

    case TransferFunction::UnknownFunctionCall:
      switch (E) {
      case LatticeElement::Maybe:
        return LatticeElement::Unknown;
      case LatticeElement::Unknown:
        return LatticeElement::Unknown;
      case LatticeElement::NoOrDead:
        return LatticeElement::NoOrDead;
      default:
        return E;
      }
      return E;

    case TransferFunction::Write:
      switch (E) {
      case LatticeElement::Maybe:
        return LatticeElement::NoOrDead;
      case LatticeElement::Unknown:
        return LatticeElement::Unknown;
      case LatticeElement::NoOrDead:
        return LatticeElement::NoOrDead;
      default:
        return E;
      }
      return E;

    default:
      return E;
    }
  }
};

} // namespace DeadReturnValuesOfFunctionCall
