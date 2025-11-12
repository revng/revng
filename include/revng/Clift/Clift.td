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

include "revng/Clift/CliftOpInterfaces.td"
include "revng/Clift/CliftOpTraits.td"
include "revng/Clift/CliftTypes.td"

class Clift_Op<string mnemonic, list<Trait> traits = []>
  : Op<Clift_Dialect,
       mnemonic,
       !listconcat(traits, [NoTerminator])>;

//===-------------------------- Type constraints --------------------------===//

def Clift_AnyLabelType
  : Type<CPred<"mlir::isa<mlir::clift::LabelType>($_self)">,
         "Clift label type",
         "mlir::clift::LabelType">,
    BuildableType<"$_builder.getType<mlir::clift::LabelType>()">;

def Clift_AnyValueType
  : Type<CPred<"llvm::isa<mlir::clift::ValueType>($_self)">,
         "Clift value type",
         "mlir::clift::ValueType">;

class Clift_AnyPrimitiveTypeOf<string Kind>
  : Type<CPred<"mlir::clift::impl::verifyPrimitiveTypeOf($_self," #
                                                         "mlir::clift::PrimitiveKind::" # Kind # ")">,
         "Clift primitive type of " # Kind,
         "mlir::clift::ValueType">;

def Clift_AnyScalarType
  : Type<CPred<"mlir::clift::isScalarType($_self)">,
         "Clift scalar type",
         "mlir::clift::ValueType">;

def Clift_AnyIntegerType
  : Type<CPred<"mlir::clift::isIntegerType($_self)">,
         "Clift integer type",
         "mlir::clift::ValueType">;

def Clift_AnyPrimitiveIntegerType
  : Type<CPred<"mlir::clift::isPrimitiveIntegerType($_self)">,
         "primitive Clift integer type",
         "mlir::clift::ValueType">;

def Clift_AnyPointerType
  : Type<CPred<"mlir::clift::isPointerType($_self)">,
         "Clift pointer type",
         "mlir::clift::ValueType">;

def Clift_AnyFunctionType
  : Type<CPred<"mlir::clift::isFunctionType($_self)">,
         "Clift function type",
         "mlir::clift::ValueType">;

class Clift_AnyNonConstType<Type type>
  : Type<And<[type.predicate,
              CPred<"not mlir::cast<mlir::clift::ValueType>($_self).isConst()">]>,
         // TODO: Figure out why the string concatenation isn't working here.
         "non-const " # type.description,
         "mlir::clift::ValueType">;

class Clift_AnyModifiableType<Type type>
  : Type<And<[type.predicate,
              CPred<"mlir::clift::isModifiableType($_self)">]>,
         // TODO: Figure out why the string concatenation isn't working here.
         "modifiable " # type.description,
         "mlir::clift::ValueType">;

class Clift_AllTypesMatch<list<string> names>
  : AllMatchSameOperatorTrait<names,
                              "mlir::clift::removeConst($_self.getType())",
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

def Clift_FunctionOp : Clift_Op<"func",
                                [Clift_GlobalOpInterface,
                                 CallableOpInterface,
                                 FunctionOpInterface,
                                 AutomaticAllocationScope,
                                 IsolatedFromAbove]> {
  let arguments = (ins StrAttr:$sym_name,
                       TypeAttrOf<Clift_AnyFunctionType>:$function_value_type,
                       OptionalAttr<DictArrayAttr>:$arg_attrs,
                       OptionalAttr<DictArrayAttr>:$res_attrs,
                       DefaultValuedStrAttr<StrAttr, "">:$handle);

  let description = [{
    expression.
  }];

  let regions = (region Clift_StatementRegion:$body);

  let builders = [
    OpBuilder<(ins "llvm::StringRef":$Name,
                   "mlir::clift::FunctionType":$FunctionType)>
  ];

  let hasVerifier = 1;
  let hasCustomAssemblyFormat = 1;

  code extraClassDeclaration = [{
    static mlir::StringAttr getFunctionTypeAttrName(mlir::OperationName Name) {
      return getFunctionValueTypeAttrName(Name);
    }

    mlir::StringAttr getFunctionTypeAttrName() {
      return getFunctionValueTypeAttrName();
    }

    mlir::TypeAttr getFunctionTypeAttr() {
      return getFunctionValueTypeAttr();
    }

    void setFunctionTypeAttr(mlir::TypeAttr Attr) {
      return setFunctionValueTypeAttr(Attr);
    }

    clift::FunctionType getFunctionType() {
      return mlir::cast<clift::FunctionType>(dealias(getFunctionValueType(),
                                                     /*IgnoreQualifiers=*/true));
    }


    clift::ValueType getReturnType() {
      return mlir::cast<clift::ValueType>(getFunctionType().getReturnType());
    }

    unsigned getArgCount() {
      return getFunctionType().getArgumentTypes().size();
    }


    mlir::DictionaryAttr getArgAttrs(unsigned Index) {
      if (auto Attrs = getArgAttrs())
        return mlir::cast<mlir::DictionaryAttr>((*Attrs)[Index]);
      return {};
    }

    mlir::Attribute getArgAttr(unsigned Index, llvm::StringRef Name) {
      if (auto Attrs = getArgAttrs(Index))
        return Attrs.get(Name);
      return {};
    }

    template<typename AttrT>
    AttrT getArgAttrOfType(unsigned Index, llvm::StringRef Name) {
      if (auto Attr = getArgAttr(Index, Name))
        return mlir::dyn_cast<AttrT>(Attr);
      return {};
    }


    mlir::DictionaryAttr getResAttrs(unsigned Index) {
      if (auto Attrs = getResAttrs())
        return mlir::cast<mlir::DictionaryAttr>((*Attrs)[Index]);
      return {};
    }

    mlir::Attribute getResAttr(unsigned Index, llvm::StringRef Name) {
      if (auto Attrs = getResAttrs(Index))
        return Attrs.get(Name);
      return {};
    }

    template<typename AttrT>
    AttrT getResAttrOfType(unsigned Index, llvm::StringRef Name) {
      if (auto Attr = getResAttr(Index, Name))
        return mlir::dyn_cast<AttrT>(Attr);
      return {};
    }


    mlir::Region::BlockListType& getBlocks() {
        return getBody().getBlocks();
    }

    Block::args_iterator args_begin() {
        return getBody().front().args_begin();
    }

    Block::args_iterator args_end() {
        return getBody().front().args_end();
    }


    // Implementation of Clift_GlobalOpInterface

    clift::ValueType getType() {
      return getFunctionType();
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
                                      [Clift_GlobalOpInterface,
                                       NoRegionArguments,
                                       IsolatedFromAbove]> {
  let arguments = (ins SymbolNameAttr:$sym_name,
                       TypeAttrOf<Clift_ValueType>:$type,
                       DefaultValuedStrAttr<StrAttr, "">:$handle);

  let description = [{
    global variable
  }];

  let regions = (region Clift_OptionalExpressionRegion:$initializer);

  let assemblyFormat = [{
    $sym_name `:` $type (`=` $initializer^)? attr-dict-with-keyword
  }];

  let hasVerifier = 1;
}

//===----------------------------- Statements -----------------------------===//

class Clift_BasicStatementOp<string mnemonic,
                             Trait interface = Clift_StatementOpInterface,
                             list<Trait> traits = []>
  : Clift_Op<mnemonic, !listconcat(traits, [interface])>;

class Clift_StatementOp<string mnemonic, list<Trait> traits = []>
  : Clift_BasicStatementOp<mnemonic,
                           Clift_StatementOpInterface,
                           !listconcat(traits, [NoRegionArguments])>;

class Clift_JumpStatementOp<string mnemonic>
  : Clift_BasicStatementOp<mnemonic,
                           Clift_JumpStatementOpInterface,
                           [NoRegionArguments, Clift_NoFallthrough]>;

// Two bit integer used to represent two flags, each indicating the presence of
// the break and continue label assignments respectively.
def LoopLabelMaskAttr : TypedUnsignedIntegerAttrBase<UI<2>,
                                                     "uint8_t",
                                                     "Loop label mask">;

class Clift_LoopStatementOp<string mnemonic, list<Trait> traits = []>
  : Clift_BasicStatementOp<mnemonic,
                           Clift_LoopOpInterface,
                           !listconcat(traits, [Clift_AssignsLoopLabels])> {

  let arguments = (ins LoopLabelMaskAttr:$label_mask,
                       Variadic<Clift_LabelType>:$labels);

  // The generation of default builders for loop operations is skipped to avoid
  // mistakes, as building a valid loop with explicit label mask and operands
  // would be error prone.
  let skipDefaultBuilders = 1;

  let builders = [
    OpBuilder<(ins CArg<"mlir::clift::LoopOpInterface", "{}">:$OtherLoop)>
  ];
}

def Clift_MakeLabelOp : Clift_StatementOp<"make_label"> {
  let arguments = (ins DefaultValuedStrAttr<StrAttr, "">:$handle);

  let results = (outs Clift_LabelType:$result);

  let assemblyFormat = [{
    attr-dict
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

  let extraClassDeclaration = [{
    [[nodiscard]] LabelAssignmentOpInterface getAssignment();
  }];
}

def Clift_AssignLabelOp
  : Clift_BasicStatementOp<"assign_label",
                           Clift_LabelAssignmentOpInterface> {

  let arguments = (ins Clift_LabelType:$label);

  let assemblyFormat = [{
    $label attr-dict
  }];

  let extraClassDeclaration = [{
    [[nodiscard]] MakeLabelOp getLabelOp();

    unsigned getAssignedLabelCount() {
      return 1;
    }

    mlir::Value getAssignedLabel(unsigned Index) {
      revng_assert(Index == 0);
      return getLabel();
    }
  }];
}

def Clift_BreakToOp : Clift_JumpStatementOp<"break_to"> {
  let arguments = (ins Clift_LabelType:$label);

  let assemblyFormat = [{
    $label attr-dict
  }];

  let hasVerifier = 1;

  let extraClassDeclaration = [{
    [[nodiscard]] MakeLabelOp getLabelOp();
  }];
}

def Clift_ContinueToOp : Clift_JumpStatementOp<"continue_to"> {
  let arguments = (ins Clift_LabelType:$label);

  let assemblyFormat = [{
    $label attr-dict
  }];

  let hasVerifier = 1;

  let extraClassDeclaration = [{
    [[nodiscard]] MakeLabelOp getLabelOp();
  }];
}

def Clift_DoWhileOp : Clift_LoopStatementOp<"do_while", [NoRegionArguments]> {
  let regions = (region Clift_StatementRegion:$body,
                        Clift_ExpressionRegion:$condition);

  let assemblyFormat = [{
    custom<CliftLoopLabels>($label_mask, $labels)
    `body` $body `cond` $condition attr-dict-with-keyword
  }];

  let hasVerifier = 1;

  let extraClassDeclaration = [{
    [[nodiscard]] bool isBooleanTestedExpression(mlir::Region &R) {
      return &R == &getCondition();
    }
  }];
}

def Clift_ExpressionStatementOp : Clift_StatementOp<"expr"> {
  let regions = (region Clift_OptionalExpressionRegion:$expression);

  let assemblyFormat = [{
    $expression attr-dict-with-keyword
  }];

  let extraClassDeclaration = [{
    [[nodiscard]] bool isDiscardedExpression(mlir::Region &R) {
      return true;
    }
  }];
}

def Clift_ForOp : Clift_LoopStatementOp<"for"> {
  let regions = (region Clift_StatementRegion:$initializer,
                        Clift_OptionalExpressionRegion:$condition,
                        Clift_OptionalExpressionRegion:$expression,
                        Clift_StatementRegion:$body);

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;

  let extraClassDeclaration = [{
    [[nodiscard]] bool isDiscardedExpression(mlir::Region &R);

    [[nodiscard]] bool isBooleanTestedExpression(mlir::Region &R) {
      return &R == &getCondition();
    }
  }];
}

def Clift_GotoOp : Clift_JumpStatementOp<"goto"> {
  let arguments = (ins Clift_LabelType:$label);

  let assemblyFormat = [{
    $label attr-dict
  }];

  let hasVerifier = 1;

  let extraClassDeclaration = [{
    [[nodiscard]] MakeLabelOp getLabelOp();
  }];
}

def Clift_IfOp : Clift_BasicStatementOp<"if", Clift_BranchOpInterface> {
  let regions = (region Clift_ExpressionRegion:$condition,
                        Clift_StatementRegion:$then,
                        Clift_StatementRegion:$else);

  let assemblyFormat = [{
    $condition `then` $then (`else` $else^)? attr-dict-with-keyword
  }];

  let hasVerifier = 1;

  let extraClassDeclaration = [{
    bool isIndirectlyNoFallthrough() const;

    [[nodiscard]] bool isBooleanTestedExpression(mlir::Region &R) {
      return &R == &getCondition();
    }
  }];
}

def Clift_LocalVariableOp : Clift_StatementOp<"local"> {
  let arguments = (ins DefaultValuedStrAttr<StrAttr, "">:$handle);

  let results = (outs Clift_AnyValueType:$result);

  let regions = (region Clift_OptionalExpressionRegion:$initializer);

  let assemblyFormat = [{
    `:` type($result) (`=` $initializer^)? attr-dict-with-keyword
  }];

  let extraClassDeclaration = [{
    mlir::clift::ValueType getType() {
      return getResult().getType();
    }
  }];

  let hasVerifier = 1;
}

def Clift_ReturnOp : Clift_StatementOp<"return", [Clift_NoFallthrough]> {

  let regions = (region Clift_OptionalExpressionRegion:$result);

  let assemblyFormat = [{
    $result attr-dict-with-keyword
  }];

  let hasVerifier = 1;
}

def Clift_SwitchOp : Clift_BasicStatementOp<"switch", Clift_BranchOpInterface> {
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
    bool isIndirectlyNoFallthrough() const;

    [[nodiscard]] mlir::clift::ValueType getConditionType();

    [[nodiscard]] Region &getConditionRegion() {
      return getRegion(0);
    }

    [[nodiscard]] bool hasDefaultCase() {
      return not getDefaultCaseRegion().empty();
    }

    [[nodiscard]] mlir::Region &getDefaultCaseRegion() {
      return getRegion(1);
    }

    [[nodiscard]] unsigned getNumCases() {
      return getNumRegions() - 2;
    }

    [[nodiscard]] uint64_t getCaseValue(const unsigned index) {
      return static_cast<uint64_t>(getCaseValues()[index]);
    }

    [[nodiscard]] llvm::MutableArrayRef<mlir::Region> getCaseRegions() {
      return getOperation()->getRegions().drop_front(2);
    }

    [[nodiscard]] mlir::Region &getCaseRegion(const unsigned index) {
      return getCaseRegions()[index];
    }

    [[nodiscard]] mlir::Region *findCaseRegion(const uint64_t value);
  }];
}

def Clift_WhileOp : Clift_LoopStatementOp<"while", [NoRegionArguments]> {
  let regions = (region Clift_ExpressionRegion:$condition,
                        Clift_StatementRegion:$body);

  let assemblyFormat = [{
    custom<CliftLoopLabels>($label_mask, $labels)
    `cond` $condition `body` $body attr-dict-with-keyword
  }];

  let hasVerifier = 1;

  let extraClassDeclaration = [{
    [[nodiscard]] bool isBooleanTestedExpression(mlir::Region &R) {
      return &R == &getCondition();
    }
  }];
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

  let extraClassDeclaration = [{
    [[nodiscard]] bool isDiscardedOperand(mlir::OpOperand &Operand);
    [[nodiscard]] bool isBooleanTestedOperand(mlir::OpOperand &Operand);
  }];
}

def Clift_ImmediateOp : Clift_ExpressionOp<"imm", [NoMemoryEffect]> {
  let arguments = (ins I64Attr:$value);

  let results = (outs Clift_AnyNonConstType<Clift_AnyIntegerType>:$result);

  let assemblyFormat = [{
    $value attr-dict `:` type($result)
  }];
}

def Clift_StringOp : Clift_ExpressionOp<"str", [NoMemoryEffect]> {
  let arguments = (ins StrAttr:$value);
  let results = (outs Clift_AnyValueType:$result);

  let assemblyFormat = [{
    $value attr-dict `:` type($result)
  }];

  let hasVerifier = 1;
}

class Clift_UnaryArithmeticOp<string mnemonic>
  : Clift_ExpressionOp<mnemonic,
                       [NoMemoryEffect,
                        TypesMatchWith<"result type must match the value type, ignoring qualifiers",
                                       "value",
                                       "result",
                                       "mlir::clift::removeConst($_self)">]> {

  let arguments = (ins Clift_AnyPrimitiveIntegerType:$value);
  let results = (outs Clift_AnyNonConstType<Clift_AnyPrimitiveIntegerType>:$result);

  let assemblyFormat = [{
    $value attr-dict `:` type($value)
  }];
}

class Clift_BinaryArithmeticOp<string mnemonic, list<Trait> traits = []>
  : Clift_ExpressionOp<mnemonic,
                       !listconcat(traits,
                                   [NoMemoryEffect,
                                    Clift_AllTypesMatch<["lhs", "rhs", "result"]>])> {

  let arguments = (ins Clift_AnyPrimitiveIntegerType:$lhs,
                       Clift_AnyPrimitiveIntegerType:$rhs);

  let results = (outs Clift_AnyNonConstType<Clift_AnyPrimitiveIntegerType>:$result);

  let builders = [
    OpBuilder<(ins "mlir::Value":$Lhs, "mlir::Value":$Rhs), [{
      build($_builder, $_state, Lhs.getType(), Lhs, Rhs);
    }]>
  ];

  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict `:`
    custom<CliftOpTypes>(type($result), type($lhs), type($rhs))
  }];
}

class Clift_PointerArithmeticOp<string mnemonic, list<Trait> traits = []>
  : Clift_ExpressionOp<mnemonic,
                       !listconcat(traits,
                                   [NoMemoryEffect])> {

  let arguments = (ins Clift_AnyValueType:$lhs, Clift_AnyValueType:$rhs);
  let results = (outs Clift_AnyNonConstType<Clift_AnyValueType>:$result);

  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict `:` `(`
    custom<CliftPointerArithmeticOpTypes>(type($result), type($lhs), type($rhs))
    `)`
  }];

  let hasVerifier = 1;

  let extraClassDeclaration = [{
    unsigned getPointerOperandIndex() {
      return mlir::clift::impl::getPointerArithmeticPointerOperandIndex(getOperation());
    }

    mlir::Value getPointer() {
      return getOperation()->getOperand(getPointerOperandIndex());
    }

    unsigned getOffsetOperandIndex() {
      return mlir::clift::impl::getPointerArithmeticOffsetOperandIndex(getOperation());
    }

    mlir::Value getOffset() {
      return getOperation()->getOperand(getOffsetOperandIndex());
    }
  }];
}

class Clift_BinaryLogicalOp<string mnemonic>
  : Clift_ExpressionOp<mnemonic,
                       [NoMemoryEffect,
                        Clift_ReturnsBoolean]> {

  let arguments = (ins Clift_AnyScalarType:$lhs,
                       Clift_AnyScalarType:$rhs);

  let results = (outs Clift_AnyNonConstType<Clift_AnyPrimitiveTypeOf<"SignedKind">>:$result);

  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict `:`
    custom<CliftOpOperandTypes>(type($lhs), type($rhs)) `->` type($result)
  }];

  let extraClassDeclaration = [{
    [[nodiscard]] bool isBooleanTestedOperand(mlir::OpOperand &Operand) {
      return true;
    }
  }];
}

class Clift_ShiftOp<string mnemonic>
  : Clift_ExpressionOp<mnemonic,
                       [NoMemoryEffect,
                        TypesMatchWith<"result type must match the lhs type, ignoring qualifiers",
                                       "lhs",
                                       "result",
                                       "mlir::clift::removeConst($_self)">]> {

  let arguments = (ins Clift_AnyPrimitiveIntegerType:$lhs,
                       Clift_AnyPrimitiveIntegerType:$rhs);

  let results = (outs Clift_AnyNonConstType<Clift_AnyPrimitiveIntegerType>:$result);

  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict `:` custom<CliftOpOperandTypes>(type($lhs), type($rhs))
  }];
}

class Clift_ComparisonOp<string mnemonic>
  : Clift_ExpressionOp<mnemonic,
                       [NoMemoryEffect,
                        Commutative,
                        Clift_ReturnsBoolean,
                        Clift_AllTypesMatch<["lhs", "rhs"]>]> {

  let arguments = (ins Clift_AnyScalarType:$lhs,
                       Clift_AnyScalarType:$rhs);

  let results = (outs Clift_AnyNonConstType<Clift_AnyPrimitiveTypeOf<"SignedKind">>:$result);

  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict `:`
    custom<CliftOpOperandTypes>(type($lhs), type($rhs)) `->` type($result)
  }];
}

class Clift_UnaryIntegerMutationOp<string mnemonic>
  : Clift_ExpressionOp<mnemonic,
                       [NoMemoryEffect,
                        AllTypesMatch<["value", "result"]>]> {

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
def Clift_AddOp : Clift_BinaryArithmeticOp<"add", [Commutative]>;
def Clift_SubOp : Clift_BinaryArithmeticOp<"sub">;
def Clift_MulOp : Clift_BinaryArithmeticOp<"mul", [Commutative]>;
def Clift_DivOp : Clift_BinaryArithmeticOp<"div">;
def Clift_RemOp : Clift_BinaryArithmeticOp<"rem">;

def Clift_PtrAddOp : Clift_PointerArithmeticOp<"ptr_add", [Commutative]>;
def Clift_PtrSubOp : Clift_PointerArithmeticOp<"ptr_sub">;

def Clift_PtrDiffOp
  : Clift_ExpressionOp<"ptr_diff",
                       [NoMemoryEffect,
                        Clift_AllTypesMatch<["lhs", "rhs"]>]> {

  let arguments = (ins Clift_AnyValueType:$lhs, Clift_AnyValueType:$rhs);
  let results = (outs Clift_AnyNonConstType<Clift_AnyValueType>:$result);

  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict `:`
    custom<CliftOpOperandTypes>(type($lhs), type($rhs)) `->` type($result)
  }];

  let hasVerifier = 1;
}

def Clift_LogicalNotOp
  : Clift_ExpressionOp<"not",
                       [NoMemoryEffect,
                        Clift_ReturnsBoolean]> {

  let arguments = (ins Clift_AnyIntegerType:$value);
  let results = (outs Clift_AnyNonConstType<Clift_AnyIntegerType>:$result);

  let assemblyFormat = [{
    $value attr-dict `:` type($value) `->` type($result)
  }];

  let extraClassDeclaration = [{
    [[nodiscard]] bool isBooleanTestedOperand(mlir::OpOperand &Operand) {
      return true;
    }
  }];
}

def Clift_LogicalAndOp : Clift_BinaryLogicalOp<"and">;
def Clift_LogicalOrOp : Clift_BinaryLogicalOp<"or">;

def Clift_BitwiseNotOp : Clift_UnaryArithmeticOp<"bitnot">;
def Clift_BitwiseAndOp : Clift_BinaryArithmeticOp<"bitand", [Commutative]>;
def Clift_BitwiseOrOp : Clift_BinaryArithmeticOp<"bitor", [Commutative]>;
def Clift_BitwiseXorOp : Clift_BinaryArithmeticOp<"bitxor", [Commutative]>;

def Clift_ShiftLeftOp : Clift_ShiftOp<"shl">;
def Clift_ShiftRightOp : Clift_ShiftOp<"shr">;

def Clift_CmpEqOp : Clift_ComparisonOp<"eq">;
def Clift_CmpNeOp : Clift_ComparisonOp<"ne">;
def Clift_CmpLtOp : Clift_ComparisonOp<"lt">;
def Clift_CmpGtOp : Clift_ComparisonOp<"gt">;
def Clift_CmpLeOp : Clift_ComparisonOp<"le">;
def Clift_CmpGeOp : Clift_ComparisonOp<"ge">;

def Clift_IncrementOp : Clift_UnaryIntegerMutationOp<"inc">;
def Clift_DecrementOp : Clift_UnaryIntegerMutationOp<"dec">;

def Clift_PostIncrementOp : Clift_UnaryIntegerMutationOp<"post_inc">;
def Clift_PostDecrementOp : Clift_UnaryIntegerMutationOp<"post_dec">;

def Clift_CastOp : Clift_ExpressionOp<"cast", [NoMemoryEffect]> {
  let arguments = (ins Clift_AnyValueType:$value,
                       Clift_CastKind:$kind);

  let results = (outs Clift_AnyNonConstType<Clift_AnyValueType>:$result);

  let assemblyFormat = [{
    `<` $kind `>` $value attr-dict `:` type($value) `->` type($result)
  }];

  let hasVerifier = 1;
}

def Clift_AddressofOp
  : Clift_ExpressionOp<"addressof",
                       [NoMemoryEffect,
                        TypesMatchWith<"object type must match the pointee type of the result type",
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
                       [NoMemoryEffect,
                        TypesMatchWith<"result type must match the pointee type of the pointer type",
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

def Clift_AccessOp : Clift_ExpressionOp<"access", [NoMemoryEffect]> {
  let arguments = (ins Clift_AnyValueType:$value,
                       UnitAttr:$indirect,
                       I64Attr:$member_index);

  let results = (outs Clift_AnyValueType:$result);

  let assemblyFormat = [{
    `<` (`indirect` $indirect^)? $member_index `>` $value attr-dict
    `:` type($value) `->` type($result)
  }];

  let extraClassDeclaration = [{
    [[nodiscard]] bool isIndirect() {
      return getIndirect();
    }

    [[nodiscard]] bool isLvalueExpression();

    [[nodiscard]] clift::ClassType getClassType();
    [[nodiscard]] clift::FieldAttr getFieldAttr();
  }];

  let hasVerifier = 1;
}

def Clift_SubscriptOp
  : Clift_ExpressionOp<"subscript",
                       [NoMemoryEffect,
                        TypesMatchWith<"result type must match the pointee type of the pointer type",
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
                                       [NoMemoryEffect,
                                        AllTypesMatch<["rhs", "result"]>]> {

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

    [[nodiscard]] bool isDiscardedOperand(mlir::OpOperand &Operand) {
      return &Operand == &getOperation()->getOpOperand(0);
    }
  }];
}

def Clift_UseOp
  : Clift_ExpressionOp<"use",
                       [NoMemoryEffect,
                        DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {

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

def Clift_TernaryOp
  : Clift_ExpressionOp<"ternary",
                       [NoMemoryEffect,
                        Clift_AllTypesMatch<["lhs", "rhs"]>,
                        TypesMatchWith<"result type must match the value type, ignoring qualifiers",
                                       "lhs",
                                       "result",
                                       "mlir::clift::removeConst($_self)">]> {

  let arguments = (ins Clift_AnyScalarType:$condition,
                       Clift_AnyValueType:$lhs,
                       Clift_AnyValueType:$rhs);

  let results = (outs Clift_AnyNonConstType<Clift_AnyValueType>:$result);

  let assemblyFormat = [{
    $condition `,` $lhs `,` $rhs attr-dict `:` `(`
    custom<CliftTernaryOpTypes>(type($condition), type($lhs), type($rhs)) `)`
  }];

  let extraClassDeclaration = [{
    [[nodiscard]] bool isBooleanTestedOperand(mlir::OpOperand &Operand) {
      return &Operand == &getOperation()->getOpOperand(0);
    }
  }];
}

def Clift_AggregateOp : Clift_ExpressionOp<"aggregate", [NoMemoryEffect]> {
  let arguments = (ins Variadic<Clift_AnyValueType>:$initializers);
  let results = (outs Clift_AnyNonConstType<Clift_AnyValueType>:$result);

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

#endif
