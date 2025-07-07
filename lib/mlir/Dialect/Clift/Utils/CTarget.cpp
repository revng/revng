//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/mlir/Dialect/Clift/Utils/CTarget.h"

using namespace mlir::clift;

const TargetCImplementation TargetCImplementation::Default = {
  .PointerSize = 8,
  .IntegerTypes = {
    { 1, clift::CIntegerKind::Char },
    { 2, clift::CIntegerKind::Short },
    { 4, clift::CIntegerKind::Int },
    { 8, clift::CIntegerKind::Long },
    { 16, clift::CIntegerKind::Extended },
  },
};
