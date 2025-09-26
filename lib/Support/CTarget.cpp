//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Support/CTarget.h"

const TargetCImplementation TargetCImplementation::Default = {
  .PointerSize = 8,
  .IntegerTypes = {
    { 1, CIntegerKind::Char },
    { 2, CIntegerKind::Short },
    { 4, CIntegerKind::Int },
    { 8, CIntegerKind::Long },
    { 16, CIntegerKind::Extended },
  },
};
