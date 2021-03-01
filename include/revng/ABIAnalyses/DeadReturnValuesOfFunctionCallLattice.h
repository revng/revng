#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// This file has been automatically generated from scripts/monotone-framework-lattice.py, please don't change it

#include "revng/Model/Binary.h"
#include "revng/ABIAnalyses/Common.h"

namespace DeadReturnValuesOfFunctionCall::CoreLattice {

using LatticeElement = model::RegisterState::Values;
using TransferFunction = ABIAnalyses::TransferKind;

inline bool isLessOrEqual(const LatticeElement &Lh, const LatticeElement &Rh) {
  return Lh == Rh
    || (Lh == LatticeElement::Maybe && Rh == LatticeElement::Unknown)
    || (Lh == LatticeElement::NoOrDead && Rh == LatticeElement::Maybe)
    || (Lh == LatticeElement::NoOrDead && Rh == LatticeElement::Unknown);
}



inline LatticeElement combineValues(const LatticeElement &Lh, const LatticeElement &Rh) {
  if ((Lh == LatticeElement::Maybe && Rh == LatticeElement::NoOrDead)
      || (Lh == LatticeElement::NoOrDead && Rh == LatticeElement::Maybe)) {
    return LatticeElement::Maybe;
  } else if ((Lh == LatticeElement::Maybe && Rh == LatticeElement::Unknown)
             || (Lh == LatticeElement::NoOrDead && Rh == LatticeElement::Unknown)
             || (Lh == LatticeElement::Unknown && Rh == LatticeElement::Maybe)
             || (Lh == LatticeElement::Unknown && Rh == LatticeElement::NoOrDead)) {
    return LatticeElement::Unknown;
  }
  return Lh;
}



inline LatticeElement transfer(TransferFunction T, const LatticeElement &E) {
  switch(T) {
  case TransferFunction::Read:
    switch(E) {
    case LatticeElement::Maybe:
      return LatticeElement::Unknown;
    case LatticeElement::NoOrDead:
      return LatticeElement::NoOrDead;
    case LatticeElement::Unknown:
      return LatticeElement::Unknown;
    default:
      return E;
    }
    return E;

  case TransferFunction::ReturnFromMaybe:
    switch(E) {
    case LatticeElement::Maybe:
      return LatticeElement::Maybe;
    case LatticeElement::NoOrDead:
      return LatticeElement::NoOrDead;
    case LatticeElement::Unknown:
      return LatticeElement::Unknown;
    default:
      return E;
    }
    return E;

  case TransferFunction::ReturnFromNoOrDead:
    switch(E) {
    case LatticeElement::Maybe:
      return LatticeElement::NoOrDead;
    case LatticeElement::NoOrDead:
      return LatticeElement::NoOrDead;
    case LatticeElement::Unknown:
      return LatticeElement::Unknown;
    default:
      return E;
    }
    return E;

  case TransferFunction::ReturnFromUnknown:
    switch(E) {
    case LatticeElement::Maybe:
      return LatticeElement::Unknown;
    case LatticeElement::NoOrDead:
      return LatticeElement::NoOrDead;
    case LatticeElement::Unknown:
      return LatticeElement::Unknown;
    default:
      return E;
    }
    return E;

  case TransferFunction::TheCall:
    switch(E) {
    case LatticeElement::Maybe:
      return LatticeElement::Unknown;
    case LatticeElement::NoOrDead:
      return LatticeElement::NoOrDead;
    case LatticeElement::Unknown:
      return LatticeElement::Unknown;
    default:
      return E;
    }
    return E;

  case TransferFunction::UnknownFunctionCall:
    switch(E) {
    case LatticeElement::Maybe:
      return LatticeElement::Unknown;
    case LatticeElement::NoOrDead:
      return LatticeElement::NoOrDead;
    case LatticeElement::Unknown:
      return LatticeElement::Unknown;
    default:
      return E;
    }
    return E;

  case TransferFunction::Write:
    switch(E) {
    case LatticeElement::Maybe:
      return LatticeElement::NoOrDead;
    case LatticeElement::NoOrDead:
      return LatticeElement::NoOrDead;
    case LatticeElement::Unknown:
      return LatticeElement::Unknown;
    default:
      return E;
    }
    return E;

  default:
    return E;
  }
}



} // namespace DeadReturnValuesOfFunctionCall::CoreLattice
