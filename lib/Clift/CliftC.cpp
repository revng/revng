//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Clift/CliftC.h"

using namespace clift;

bool clift::equivalentInC(mlir::Type LHS, mlir::Type RHS) {
  if (LHS == RHS)
    return true;

  if (auto LI = mlir::dyn_cast<IntegerType>(LHS)) {
    if (auto RI = mlir::dyn_cast<IntegerType>(RHS)) {

      static constexpr IntegerKind Signed = IntegerKind::Signed;

      // Primitive integer types of the same size and signedness are equivalent.
      if (LI.getSize() == RI.getSize()
          and (LI.getKind() == Signed) == (RI.getKind() == Signed)
          and LI.getIsConst() == RI.getIsConst())
        return true;
    }
  }

  return false;
}

bool clift::isNullPointerConstantInC(mlir::Value Value) {
  if (Value.getDefiningOp<NullOp>())
    return true;

  if (auto Immediate = Value.getDefiningOp<ImmediateOp>()) {
    return Immediate.getValue().isZero()
           and mlir::isa<IntegerType>(Value.getType());
  }

  return false;
}

//===------------------------ Implicit conversions ------------------------===//

static bool isImplicitPointerConversionInC(mlir::Type Source,
                                           mlir::Type Target,
                                           const CDialect &Dialect) {
  Source = collapseTypedefs(Source);
  Target = collapseTypedefs(Target);

  // Conversions to and from function pointer types are not implicit.
  if (mlir::isa<FunctionType>(Source) or mlir::isa<FunctionType>(Target))
    return false;

  // Conversion which remove qualifiers may not be implicit, depending on
  // configuration.
  if (not Dialect.ImplicitQualifierDiscardingConversions and isConst(Source)
      and not isConst(Target))
    return false;

  Source = removeConst(Source);
  Target = removeConst(Target);

  // Otherwise, conversions between pointers with equivalent pointee types are
  // implicit.
  if (equivalentInC(Source, Target))
    return true;

  // Conversions to void pointer are implicit.
  if (mlir::isa<VoidType>(Target))
    return true;

  // Conversions from void pointer may be implicit, depending on configuration.
  if (Dialect.ImplicitVoidPointerConversions and mlir::isa<VoidType>(Source))
    return true;

  if (Dialect.ImplicitIncompatiblePointerConversions)
    return true;

  // Conversion between integer or C character types of equal width may be
  // implicit, depending on configuration.
  if (Dialect.ImplicitPointerSignConversions) {
    if (mlir::isa<IntegerType>(Target)
        and mlir::isa<IntegerType, CCharType>(Source)
        and getObjectSize(Source) == getObjectSize(Target))
      return true;
  }

  // No other conversion between pointer types is implicit.
  return false;
}

bool clift::isImplicitlyConvertibleInC(mlir::Type Source,
                                       mlir::Type Target,
                                       const CDialect &Dialect) {
  Source = collapseTypedefs(Source);
  Target = collapseTypedefs(Target);

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

      if (isImplicitPointerConversionInC(SP.getPointeeType(),
                                         TP.getPointeeType(),
                                         Dialect))
        return true;
    }

    // Function-to-pointer decay conversions are implicit.
    if (mlir::isa<FunctionType>(Source) and TP.getPointeeType() == Source)
      return true;

    // Array-to-pointer decay conversions are implicit, iff the equivalent
    // pointer conversion is implicit.
    if (auto SA = mlir::dyn_cast<ArrayType>(Source))
      return isImplicitPointerConversionInC(SA.getElementType(),
                                            TP.getPointeeType(),
                                            Dialect);
  }

  return false;
}

bool clift::isImplicitConversionInC(CastOpInterface Cast,
                                    const CDialect &Dialect) {
  CDialect LocalDialect = Dialect;

  // Conversion from void pointer to another pointer type is always considered
  // implicit when the converted expression is a null pointer constant.
  if (mlir::isa<BitCastOp>(Cast) and isNullPointerConstantInC(Cast.getValue()))
    LocalDialect.ImplicitVoidPointerConversions = true;

  return isImplicitlyConvertibleInC(Cast.getValueType(),
                                    Cast.getType(),
                                    LocalDialect);
}
