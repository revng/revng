//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Clift/CliftC.h"

using namespace clift;

//===------------------------ Implicit conversions ------------------------===//

static bool isNullPointerConstant(mlir::Value Value) {
  if (Value.getDefiningOp<NullOp>())
    return true;

  if (auto Immediate = Value.getDefiningOp<ImmediateOp>()) {
    return Immediate.getValue().isZero()
           and mlir::isa<IntegerType>(Value.getType());
  }

  return false;
}

static bool isImplicitPointerConversion(mlir::Type Source, mlir::Type Target) {
  Source = collapseTypedefs(Source);
  Target = collapseTypedefs(Target);

  // Conversions to and from function pointer types are not implicit.
  if (mlir::isa<FunctionType>(Source) or mlir::isa<FunctionType>(Target))
    return false;

  // Conversion which remove qualifiers are not implicit.
  if (isConst(Source) and not isConst(Target))
    return false;

  // Otherwise, conversions between pointers with equivalent pointee
  // types are implicit.
  if (equivalent(Source, Target))
    return true;

  // Conversion to and from void pointers are implicit.
  if (mlir::isa<VoidType>(Source) or mlir::isa<VoidType>(Target))
    return true;

  // No other conversion between pointer types is implicit.
  return false;
}

bool clift::c::isImplicitlyConvertible(mlir::Type Source, mlir::Type Target) {
  Source = unwrapTypedefs(Source);
  Target = unwrapTypedefs(Target);

  // All conversions between boolean and integer types are implicit.
  if (mlir::isa<BoolType, IntegralType>(Source)
      and mlir::isa<BoolType, IntegralType>(Target))
    return true;

  // Conversions from any scalar type to boolean are implicit.
  if (isScalarType(Source) and mlir::isa<BoolType>(Target))
    return true;

  if (auto TP = mlir::dyn_cast<PointerType>(Target)) {
    if (auto SP = mlir::dyn_cast<PointerType>(Source)) {

      // Conversions between differently sized pointer types are not implicit.
      if (SP.getPointerSize() != TP.getPointerSize())
        return false;

      if (isImplicitPointerConversion(SP.getPointeeType(), TP.getPointeeType()))
        return true;
    }

    // Function-to-pointer decay conversions are implicit.
    if (mlir::isa<FunctionType>(Source) and TP.getPointeeType() == Source)
      return true;

    // Array-to-pointer decay conversions are implicit, iff the equivalent
    // pointer conversion is implicit.
    if (auto SA = mlir::dyn_cast<ArrayType>(Source))
      return isImplicitPointerConversion(SA.getElementType(),
                                         TP.getPointeeType());
  }

  return false;
}

bool clift::c::isImplicitConversion(CastOpInterface Cast) {
  mlir::Type Target = unwrapTypedefs(Cast.getType());

  if (mlir::isa<BitCastOp>(Cast)) {
    // Bitwise conversion from null pointer constants to any pointer type are
    // implicit.
    if (mlir::isa<PointerType>(Target)
        and isNullPointerConstant(Cast.getValue()))
      return true;
  }

  mlir::Type Source = unwrapTypedefs(Cast.getValueType());

  return isImplicitlyConvertible(Source, Target);
}
