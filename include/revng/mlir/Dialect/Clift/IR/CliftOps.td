//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#ifndef MLIR_CLIFT_OPS
#define MLIR_CLIFT_OPS

include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/FunctionInterfaces.td"
include "mlir/IR/RegionKindInterface.td"
include "mlir/IR/SymbolInterfaces.td"

include "CliftOpInterfaces.td"
include "CliftOpTraits.td"
include "CliftTypes.td"

class Clift_Op<string mnemonic, list<Trait> traits = []>
  : Op<Clift_Dialect,
       mnemonic,
       !listconcat(traits, [NoTerminator])>;

//===-------------------------- Type constraints --------------------------===//

def Clift_AnyValueType
  : Type<CPred<"llvm::isa<mlir::clift::ValueType>($_self)">,
         "a Clift value type",
         "mlir::clift::ValueType">;

class Clift_AnyPrimitiveTypeOf<string Kind>
  : Type<CPred<"mlir::clift::impl::verifyPrimitiveTypeOf($_self," #
                                                         "mlir::clift::PrimitiveKind::" # Kind # ")">,
         "a Clift primitive type of " # Kind,
         "mlir::clift::PrimitiveType">;

def Clift_AnyScalarType
  : Type<CPred<"mlir::clift::isScalarType($_self)">,
         "a Clift scalar type",
         "mlir::clift::ValueType">;

def Clift_AnyIntegerType
  : Type<CPred<"mlir::clift::isIntegerType($_self)">,
         "a Clift integer type",
         "mlir::clift::ValueType">;

def Clift_AnyPrimitiveIntegerType
  : Type<CPred<"mlir::clift::isPrimitiveIntegerType($_self)">,
         "a primitive Clift integer type",
         "mlir::clift::PrimitiveType">;

def Clift_AnyPointerType
  : Type<CPred<"mlir::clift::isPointerType($_self)">,
         "a Clift pointer type",
         "mlir::clift::PointerType">;

def Clift_AnyFunctionType
  : Type<CPred<"mlir::clift::isFunctionType($_self)">,
         "a Clift function type",
         "mlir::clift::ValueType">;

class Clift_AnyNonConstType<Type type>
  : Type<And<[type.predicate,
              CPred<"not mlir::cast<mlir::clift::ValueType>($_self).isConst()">]>,
         "non-const",
         "mlir::clift::ValueType">;

class Clift_AnyModifiableType<Type type>
  : Type<And<[type.predicate,
              CPred<"mlir::clift::isModifiableType($_self)">]>,
         "modifiable",
         "mlir::clift::ValueType">;

class Clift_AllTypesMatch<list<string> names>
  : AllMatchSameOperatorTrait<names,
                              "mlir::clift::impl::removeCliftConst($_self.getType())",
                              "type, ignoring qualifiers">;

//===---------------------------- Region types ----------------------------===//

def Clift_StatementRegion
  : Region<CPred<"::mlir::clift::impl::verifyStatementRegion($_self)">,
           "Region containing statements">;

def Clift_ExpressionRegion
  : Region<CPred<"::mlir::clift::impl::verifyExpressionRegion($_self, true)">,
           "Region representing an expression">;

def Clift_OptionalExpressionRegion
  : Region<CPred<"::mlir::clift::impl::verifyExpressionRegion($_self, false)">,
           "Optional region representing an expression">;

//===------------------------ Top level operations ------------------------===//

def Clift_ModuleOp : Clift_Op<"module",
                              [SymbolTable,
                               HasOnlyGraphRegion,
                               NoRegionArguments,
                               SingleBlock,
                               RegionKindInterface,
                               IsolatedFromAbove]> {
  let regions = (region AnyRegion:$body);

  let skipDefaultBuilders = 1;
  let builders = [OpBuilder<(ins)>];

  let assemblyFormat = [{
    $body attr-dict
  }];

  let hasVerifier = 1;
}

def Clift_FunctionOp : Clift_Op<"func",
                                [Symbol,
                                 CallableOpInterface,
                                 FunctionOpInterface,
                                 AutomaticAllocationScope,
                                 IsolatedFromAbove]> {
  let arguments = (ins StrAttr:$sym_name,
                       TypeAttrOf<Clift_AnyFunctionType>:$function_type,
                       OptionalAttr<DictArrayAttr>:$arg_attrs,
                       OptionalAttr<DictArrayAttr>:$res_attrs,
                       DefaultValuedStrAttr<StrAttr, "">:$unique_handle);

  let description = [{
    expression.
  }];

  let regions = (region Clift_StatementRegion:$body);

  let hasCustomAssemblyFormat = 1;

  code extraClassDeclaration = [{
    mlir::Region::BlockListType& getBlocks() {
        return getBody().getBlocks();
    }

    Block::args_iterator args_begin() {
        return getBody().front().args_begin();
    }

    Block::args_iterator args_end() {
        return getBody().front().args_end();
    }


    // Implementation of CallableOpInterface

    Region *getCallableRegion() {
        return isExternal() ? nullptr : &getBody();
    }

    /// Returns the result types of this function.
    ArrayRef<Type> getCallableResults() {
      return getResultTypes();
    }


    // Implementation of FunctionOpInterface

    ArrayRef<Type> getArgumentTypes();
    ArrayRef<Type> getResultTypes();

    Type cloneTypeWith(TypeRange inputs, TypeRange results);
  }];
}

def Clift_GlobalVariableOp : Clift_Op<"global",
                                      [Symbol,
                                       NoRegionArguments,
                                       IsolatedFromAbove]> {
  let arguments = (ins SymbolNameAttr:$sym_name,
                       TypeAttrOf<Clift_ValueType>:$type,
                       DefaultValuedStrAttr<StrAttr, "">:$unique_handle);

  let description = [{
    global variable
  }];

  let regions = (region Clift_OptionalExpressionRegion:$initializer);

  let assemblyFormat = [{
    $type $sym_name (`=` $initializer^)? attr-dict
  }];

  let hasVerifier = 1;
}

//===----------------------------- Statements -----------------------------===//

class Clift_StatementOp<string mnemonic, list<Trait> traits = []>
  : Clift_Op<mnemonic,
             !listconcat(traits,
                         [Clift_StatementOpInterface,
                          NoRegionArguments])>;

def Clift_MakeLabelOp
  : Clift_StatementOp<"make_label",
                      [Clift_OneUseOfType<"mlir::clift::AssignLabelOp">]> {

  let results = (outs Clift_LabelType:$result);

  let assemblyFormat = [{
    type($result) attr-dict
  }];

  let hasCanonicalizeMethod = 1;
  let hasVerifier = 1;

  let builders = [
    OpBuilder<(ins), [{
      build($_builder,
            $_state,
            mlir::clift::LabelType::get($_builder.getContext()));
    }]>
  ];
}

def Clift_AssignLabelOp : Clift_StatementOp<"assign_label"> {
  let arguments = (ins Clift_LabelType:$label);

  let assemblyFormat = [{
    $label type($label) attr-dict
  }];
}

def Clift_DoWhileOp : Clift_StatementOp<"do_while"> {
  let regions = (region Clift_StatementRegion:$body,
                        Clift_ExpressionRegion:$condition);

  let assemblyFormat = [{
    $body $condition attr-dict
  }];

  let hasVerifier = 1;
}

def Clift_ExpressionStatementOp : Clift_StatementOp<"expr"> {
  let regions = (region Clift_OptionalExpressionRegion:$expression);

  let assemblyFormat = [{
    $expression attr-dict
  }];
}

def Clift_ForOp : Clift_StatementOp<"for"> {
  let regions = (region Clift_StatementRegion:$initializer,
                        Clift_OptionalExpressionRegion:$condition,
                        Clift_OptionalExpressionRegion:$expression,
                        Clift_StatementRegion:$body);

  let assemblyFormat = [{
    $initializer $condition $expression $body attr-dict
  }];

  let hasVerifier = 1;
}

def Clift_GoToOp : Clift_StatementOp<"goto", [Clift_NoFallthrough]> {
  let arguments = (ins Clift_LabelType:$label);

  let assemblyFormat = [{
    $label type($label) attr-dict
  }];
}

def Clift_IfOp : Clift_StatementOp<"if"> {
  let regions = (region Clift_ExpressionRegion:$condition,
                        Clift_StatementRegion:$then,
                        Clift_StatementRegion:$else);

  let assemblyFormat = [{
    $condition $then (`else` $else^)? attr-dict
  }];

  let hasVerifier = 1;
}

def Clift_LocalVariableOp : Clift_StatementOp<"local"> {
  let arguments = (ins StrAttr:$sym_name);

  let results = (outs Clift_AnyValueType:$result);

  let regions = (region Clift_OptionalExpressionRegion:$initializer);

  let assemblyFormat = [{
    type($result) $sym_name (`=` $initializer^)? attr-dict
  }];

  let extraClassDeclaration = [{
    mlir::clift::ValueType getType() {
      return getResult().getType();
    }
  }];

  let hasVerifier = 1;
}

def Clift_LoopBreakOp : Clift_StatementOp<"loop_break", [Clift_NoFallthrough]> {
  let assemblyFormat = [{
    attr-dict
  }];
}

def Clift_LoopContinueOp : Clift_StatementOp<"loop_continue",
                                             [Clift_NoFallthrough]> {

  let assemblyFormat = [{
    attr-dict
  }];
}

def Clift_ReturnOp : Clift_StatementOp<"return", [Clift_NoFallthrough]> {

  let regions = (region Clift_OptionalExpressionRegion:$result);

  let assemblyFormat = [{
    $result attr-dict
  }];

  let hasVerifier = 1;
}

def Clift_SwitchOp : Clift_StatementOp<"switch"> {
  let arguments = (ins DenseI64ArrayAttr:$case_values);

  let regions = (region Clift_ExpressionRegion:$condition,
                        Clift_StatementRegion:$default,
                        VariadicRegion<Clift_StatementRegion>:$cases);

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;

  let builders = [
    OpBuilder<(ins "llvm::ArrayRef<uint64_t>":$case_values)>
  ];

  code extraClassDeclaration = [{
    [[nodiscard]] Region &getConditionRegion() {
      return getRegion(0);
    }

    [[nodiscard]] bool hasDefaultCase() {
      return not getDefaultCaseRegion().empty();
    }

    [[nodiscard]] Region &getDefaultCaseRegion() {
      return getRegion(1);
    }

    [[nodiscard]] unsigned getNumCases() {
      return getNumRegions() - 2;
    }

    [[nodiscard]] uint64_t getCaseValue(const unsigned index) {
      return static_cast<uint64_t>(getCaseValues()[index]);
    }

    [[nodiscard]] Region &getCaseRegion(const unsigned index) {
      return getRegion(index + 2u);
    }

    [[nodiscard]] Region *findCaseRegion(const uint64_t value);
  }];
}

def Clift_SwitchBreakOp : Clift_StatementOp<"switch_break",
                                            [Clift_NoFallthrough]> {

  let assemblyFormat = [{
    attr-dict
  }];
}

def Clift_WhileOp : Clift_StatementOp<"while"> {
  let regions = (region Clift_ExpressionRegion:$condition,
                        Clift_StatementRegion:$body);

  let assemblyFormat = [{
    $condition $body attr-dict
  }];

  let hasVerifier = 1;
}

//===----------------------------- Expressions ----------------------------===//

class Clift_ExpressionOp<string mnemonic, list<Trait> traits = []>
  : Clift_Op<mnemonic,
             !listconcat(traits,
                         [Clift_ExpressionOpInterface,
                          NoRegionArguments])>;

def Clift_UndefOp : Clift_ExpressionOp<"undef"> {
  let results = (outs Clift_AnyValueType:$result);

  let assemblyFormat = [{
    attr-dict `:` type($result)
  }];
}

def Clift_YieldOp : Clift_ExpressionOp<"yield", [Terminator]> {
  let arguments = (ins Clift_AnyValueType:$value);

  let assemblyFormat = [{
    $value attr-dict `:` type($value)
  }];
}

def Clift_ImmediateOp : Clift_ExpressionOp<"imm"> {
  let arguments = (ins I64Attr:$value);

  let results = (outs Clift_AnyNonConstType<Clift_AnyIntegerType>:$result);

  let assemblyFormat = [{
    $value attr-dict `:` type($result)
  }];
}

class Clift_UnaryArithmeticOp<string mnemonic>
  : Clift_ExpressionOp<mnemonic,
                       [TypesMatchWith<"result type must match the value type, ignoring qualifiers",
                                       "value",
                                       "result",
                                       "mlir::clift::impl::removeCliftConst($_self)">]> {

  let arguments = (ins Clift_AnyPrimitiveIntegerType:$value);
  let results = (outs Clift_AnyNonConstType<Clift_AnyPrimitiveIntegerType>:$result);

  let assemblyFormat = [{
    $value attr-dict `:` type($value)
  }];
}

class Clift_BinaryArithmeticOp<string mnemonic>
  : Clift_ExpressionOp<mnemonic,
                       [Clift_AllTypesMatch<["lhs", "rhs", "result"]>]> {

  let arguments = (ins Clift_AnyPrimitiveIntegerType:$lhs,
                       Clift_AnyPrimitiveIntegerType:$rhs);

  let results = (outs Clift_AnyNonConstType<Clift_AnyPrimitiveIntegerType>:$result);

  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict `:`
    custom<CliftOpTypes>(type($result), type($lhs), type($rhs))
  }];
}

class Clift_BinaryLogicalOp<string mnemonic> : Clift_ExpressionOp<mnemonic> {
  let arguments = (ins Clift_AnyPrimitiveIntegerType:$lhs,
                       Clift_AnyPrimitiveIntegerType:$rhs);

  let results = (outs Clift_AnyNonConstType<Clift_AnyPrimitiveTypeOf<"SignedKind">>:$result);

  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict `:`
    custom<CliftOpOperandTypes>(type($lhs), type($rhs)) `->` type($result)
  }];
}

class Clift_ShiftOp<string mnemonic>
  : Clift_ExpressionOp<mnemonic,
                       [TypesMatchWith<"result type must match the lhs type, ignoring qualifiers",
                                       "lhs",
                                       "result",
                                       "mlir::clift::impl::removeCliftConst($_self)">]> {

  let arguments = (ins Clift_AnyPrimitiveIntegerType:$lhs,
                       Clift_AnyPrimitiveIntegerType:$rhs);

  let results = (outs Clift_AnyNonConstType<Clift_AnyPrimitiveIntegerType>:$result);

  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict `:` custom<CliftOpOperandTypes>(type($lhs), type($rhs))
  }];
}

class Clift_ComparisonOp<string mnemonic>
  : Clift_ExpressionOp<mnemonic,
                       [Clift_AllTypesMatch<["lhs", "rhs"]>]> {

  let arguments = (ins Clift_AnyScalarType:$lhs,
                       Clift_AnyScalarType:$rhs);

  let results = (outs Clift_AnyNonConstType<Clift_AnyPrimitiveTypeOf<"SignedKind">>:$result);

  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict `:`
    custom<CliftOpTypes>(type($result), type($lhs), type($rhs))
  }];
}

class Clift_UnaryIntegerMutationOp<string mnemonic>
  : Clift_ExpressionOp<mnemonic,
                       [AllTypesMatch<["value", "result"]>]> {

  let arguments = (ins Clift_AnyModifiableType<Clift_AnyPrimitiveIntegerType>:$value);
  let results = (outs Clift_AnyPrimitiveIntegerType:$result);

  let assemblyFormat = [{
    $value attr-dict `:` type($value)
  }];

  let extraClassDeclaration = [{
    bool isLvalueExpression() {
      return true;
    }

    mlir::LogicalResult verify() {
      return mlir::clift::impl::verifyUnaryIntegerMutationOp(getOperation());
    }
  }];
}

def Clift_NegOp : Clift_UnaryArithmeticOp<"neg">;
def Clift_AddOp : Clift_BinaryArithmeticOp<"add">;
def Clift_SubOp : Clift_BinaryArithmeticOp<"sub">;
def Clift_MulOp : Clift_BinaryArithmeticOp<"mul">;
def Clift_DivOp : Clift_BinaryArithmeticOp<"div">;
def Clift_RemOp : Clift_BinaryArithmeticOp<"rem">;

def Clift_LogicalNotOp : Clift_ExpressionOp<"not"> {
  let arguments = (ins Clift_AnyIntegerType:$value);
  let results = (outs Clift_AnyNonConstType<Clift_AnyIntegerType>:$result);

  let assemblyFormat = [{
    $value attr-dict `:` type($value) `->` type($result)
  }];
}

def Clift_LogicalAndOp : Clift_BinaryLogicalOp<"and">;
def Clift_LogicalOrOp : Clift_BinaryLogicalOp<"or">;

def Clift_BitwiseNotOp : Clift_UnaryArithmeticOp<"bitnot">;
def Clift_BitwiseAndOp : Clift_BinaryArithmeticOp<"bitand">;
def Clift_BitwiseOrOp : Clift_BinaryArithmeticOp<"bitor">;
def Clift_BitwiseXorOp : Clift_BinaryArithmeticOp<"bitxor">;

def Clift_ShiftLeftOp : Clift_ShiftOp<"shl">;
def Clift_ShiftRightOp : Clift_ShiftOp<"shr">;

def Clift_EqualOp : Clift_ComparisonOp<"eq">;
def Clift_NotEqualOp : Clift_ComparisonOp<"ne">;
def Clift_LessThanOp : Clift_ComparisonOp<"lt">;
def Clift_GreaterThanOp : Clift_ComparisonOp<"gt">;
def Clift_LessThanOrEqual : Clift_ComparisonOp<"le">;
def Clift_GreaterThanOrEqual : Clift_ComparisonOp<"ge">;

def Clift_IncrementOp : Clift_UnaryIntegerMutationOp<"inc">;
def Clift_DecrementOp : Clift_UnaryIntegerMutationOp<"dec">;

def Clift_PostIncrementOp : Clift_UnaryIntegerMutationOp<"post_inc">;
def Clift_PostDecrementOp : Clift_UnaryIntegerMutationOp<"post_dec">;

def Clift_CastOp : Clift_ExpressionOp<"cast"> {
  let arguments = (ins Clift_AnyValueType:$value,
                       Clift_CastKind:$kind,
                       DefaultValuedStrAttr<StrAttr, "">:$purpose);

  let results = (outs Clift_AnyNonConstType<Clift_AnyValueType>:$result);

  let assemblyFormat = [{
    `<` $kind `>` $value attr-dict `:` type($value) `->` type($result)
  }];

  let hasVerifier = 1;
}

def Clift_AddressofOp
  : Clift_ExpressionOp<"addressof",
                       [TypesMatchWith<"object type must match the pointee type of the result type",
                                       "result",
                                       "object",
                                       "mlir::cast<PointerType>($_self).getPointeeType()">]> {

  let arguments = (ins Clift_AnyValueType:$object);
  let results = (outs Clift_AnyNonConstType<Clift_AnyPointerType>:$result);

  let assemblyFormat = [{
    $object attr-dict `:` type($result)
  }];

  let hasVerifier = 1;
}

def Clift_IndirectionOp
  : Clift_ExpressionOp<"indirection",
                       [TypesMatchWith<"result type must match the pointee type of the pointer type",
                                       "pointer",
                                       "result",
                                       "mlir::cast<PointerType>($_self).getPointeeType()">]> {

  let arguments = (ins Clift_AnyPointerType:$pointer);
  let results = (outs Clift_AnyValueType:$result);

  let assemblyFormat = [{
    $pointer attr-dict `:` type($pointer)
  }];

  let extraClassDeclaration = [{
    bool isLvalueExpression() {
      return true;
    }
  }];

  let hasVerifier = 1;
}

def Clift_AssignOp
  : Clift_ExpressionOp<"assign",
                       [AllTypesMatch<["lhs", "result"]>,
                        Clift_AllTypesMatch<["lhs", "rhs"]>]> {

  let arguments = (ins Clift_AnyModifiableType<Clift_AnyValueType>:$lhs,
                       Clift_AnyValueType:$rhs);

  let results = (outs Clift_AnyValueType:$result);

  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict `:`
    custom<CliftOpOperandTypes>(type($lhs), type($rhs))
  }];

  let extraClassDeclaration = [{
    bool isLvalueExpression() {
      return true;
    }
  }];

  let hasVerifier = 1;
}

def Clift_ObjectAccessOp : Clift_ExpressionOp<"object_access"> {
  let arguments = (ins Clift_AnyValueType:$object,
                       I64Attr:$member_index);

  let results = (outs Clift_AnyValueType:$result);

  let assemblyFormat = [{
    `<` $member_index `>` $object attr-dict `:` type($object) `->` type($result)
  }];

  let extraClassDeclaration = [{
    bool isLvalueExpression();
  }];

  let hasVerifier = 1;
}

def Clift_PointerAccessOp : Clift_ExpressionOp<"pointer_access"> {
  let arguments = (ins Clift_AnyPointerType:$pointer,
                       I64Attr:$member_index);

  let results = (outs Clift_AnyValueType:$result);

  let assemblyFormat = [{
    `<` $member_index `>` $pointer attr-dict `:` type($pointer) `->` type($result)
  }];

  let extraClassDeclaration = [{
    bool isLvalueExpression() {
      return true;
    }
  }];

  let hasVerifier = 1;
}

def Clift_SubscriptOp
  : Clift_ExpressionOp<"subscript",
                       [TypesMatchWith<"result type must match the pointee type of the pointer type",
                                       "pointer",
                                       "result",
                                       "mlir::cast<PointerType>($_self).getPointeeType()">]> {
  let arguments = (ins Clift_AnyValueType:$pointer,
                       Clift_AnyIntegerType:$index);

  let results = (outs Clift_AnyValueType:$result);

  let assemblyFormat = [{
    $pointer `,` $index attr-dict `:` `(` type($pointer) `,` type($index) `)`
  }];

  let extraClassDeclaration = [{
    bool isLvalueExpression() {
      return true;
    }
  }];

  let hasVerifier = 1;
}

def Clift_CommaOp : Clift_ExpressionOp<"comma",
                                       [AllTypesMatch<["rhs", "result"]>]> {

  let arguments = (ins Clift_AnyValueType:$lhs,
                       Clift_AnyValueType:$rhs);
  let results = (outs Clift_AnyValueType:$result);

  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs)
  }];

  let extraClassDeclaration = [{
    bool isLvalueExpression() {
      return mlir::clift::isLvalueExpression(getRhs());
    }
  }];
}

def Clift_UseOp
  : Clift_ExpressionOp<"use",
                       [DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {

  let arguments = (ins FlatSymbolRefAttr:$symbol_name);
  let results = (outs Clift_AnyValueType:$result);

  let assemblyFormat = [{
    $symbol_name attr-dict `:` type($result)
  }];

  let extraClassDeclaration = [{
    bool isLvalueExpression() {
      return true;
    }
  }];
}

def Clift_CallOp : Clift_ExpressionOp<"call"> {
  let arguments = (ins Clift_AnyValueType:$function,
                       Variadic<Clift_AnyValueType>:$arguments);

  let results = (outs Clift_AnyNonConstType<Clift_AnyValueType>:$result);

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

#endif
