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

def Clift_AnyAddressableType
  : Type<CPred<"mlir::isa<clift::AddressableType>($_self)">,
         "value type">;

def Clift_AnyObjectType
  : Type<CPred<"clift::unwrapped_isa<clift::ObjectType>($_self)">,
         "object type">;

def Clift_AnyValueType
  : Type<CPred<"clift::unwrapped_isa<clift::ValueType>($_self)">,
         "value type">;

def Clift_AnyBooleanType
  : Type<CPred<"clift::isBooleanType($_self)">,
         "boolean type">;

def Clift_AnyScalarType
  : Type<CPred<"clift::isScalarType($_self)">,
         "scalar type">;

def Clift_AnyIntegerType
  : Type<CPred<"clift::unwrapped_isa<clift::IntegralType>($_self)">,
         "integer type">;

def Clift_AnyPrimitiveIntegerType
  : Type<CPred<"clift::unwrapped_isa<clift::IntegerType>($_self)">,
         "primitive integer type">;

def Clift_AnyPointerType
  : Type<CPred<"clift::unwrapped_isa<clift::PointerType>($_self)">,
         "pointer type">;

def Clift_AnyFunctionType
  : Type<CPred<"clift::unwrapped_isa<clift::FunctionType>($_self)">,
         "function type">;

def Clift_AnyNonVoidType
  : Type<CPred<"not clift::unwrapped_isa<clift::VoidType>($_self)">,
         "non-void type">;

class Clift_AnyNonConstType<Type type>
  : Type<And<[type.predicate,
              CPred<"not clift::isConst($_self)">]>,
         "non-const " # type.summary>;

class Clift_AnyModifiableType<Type type>
  : Type<And<[type.predicate,
              CPred<"clift::isModifiableType($_self)">]>,
         "modifiable " # type.summary>;

class Clift_AllTypesMatch<list<string> names>
  : AllMatchSameOperatorTrait<names,
                              "clift::removeConst($_self.getType())",
                              "type (ignoring qualifiers)">;

class Clift_AllSizesMatch<list<string> names>
  : AllMatchSameOperatorTrait<names,
                              "clift::getObjectSize($_self.getType())",
                              "object size">;

class Clift_ValuesAreOrdered<string lhs_name, string rhs_name, string operator, string summary>
  : PredOpTrait<"the " # summary # " of " # lhs_name # " and " # rhs_name # " are ordered",
                CPred<!subst("$_self", "$" # lhs_name, operator) # " < " # !subst("$_self", "$" # rhs_name, operator)>>;

class Clift_ObjectSizesAreOrdered<string lhs_name, string rhs_name>
  : Clift_ValuesAreOrdered<lhs_name, rhs_name, "clift::getObjectSize($_self.getType())", "sizes">;

class Clift_AllPointeeTypesMatch<list<string> names>
  : AllMatchSameOperatorTrait<names,
                              "clift::unwrapped_cast<clift::PointerType>($_self.getType()).getPointeeType()",
                              "pointee type">;

class Clift_UnqualifiedTypeMatches<string qualified, string unqualified>
  : TypesMatchWith<qualified # " type (ignoring qualifiers) matches the " # unqualified # " type",
                   qualified,
                   unqualified,
                   "clift::removeConst($_self)">;

class Clift_PointeeTypeMatches<string pointer, string value>
  : TypesMatchWith<pointer # " pointee type matches the " # value # " type",
                   pointer,
                   value,
                   "clift::unwrapped_cast<clift::PointerType>($_self).getPointeeType()">;

class Clift_LValueOperand<string name>
  : PredOpTrait<"operand " # name # " is an lvalue expression",
                CPred<"clift::isLvalueExpression($" # name # ")">>;

//===---------------------------- Region types ----------------------------===//

def Clift_StatementRegion
  : Region<CPred<"clift::impl::verifyStatementRegion($_self)">,
           "Region containing statements">;

def Clift_ExpressionRegion
  : Region<CPred<"clift::impl::verifyExpressionRegion($_self, true)">,
           "Region representing an expression">;

def Clift_OptionalExpressionRegion
  : Region<CPred<"clift::impl::verifyExpressionRegion($_self, false)">,
           "Optional region representing an expression">;

//===------------------------ Top level operations ------------------------===//

def Clift_FunctionOp : Clift_Op<"func",
                                [Clift_GlobalOpInterface,
                                 CallableOpInterface,
                                 FunctionOpInterface,
                                 AutomaticAllocationScope,
                                 IsolatedFromAbove]> {
  let arguments = (ins StrAttr:$sym_name,
                       TypeAttrOf<Clift_AnyFunctionType>:$type,
                       OptionalAttr<DictArrayAttr>:$arg_attrs,
                       OptionalAttr<DictArrayAttr>:$res_attrs,
                       UnitAttr:$noreturn,
                       UnitAttr:$always_inline,
                       UnitAttr:$has_one_broken_return,
                       DefaultValuedStrAttr<StrAttr, "">:$handle);

  let description = [{
    expression.
  }];

  let regions = (region Clift_StatementRegion:$body);

  let builders = [
    OpBuilder<(ins "llvm::StringRef":$Name,
                   "clift::FunctionType":$FunctionType)>
  ];

  let hasCustomAssemblyFormat = 1;

  code extraClassDeclaration = [{
    static mlir::StringAttr getFunctionTypeAttrName(mlir::OperationName Name) {
      return getTypeAttrName(Name);
    }

    mlir::StringAttr getFunctionTypeAttrName() {
      return getTypeAttrName();
    }

    mlir::TypeAttr getFunctionTypeAttr() {
      return getTypeAttr();
    }

    void setFunctionTypeAttr(mlir::TypeAttr Attr) {
      return setTypeAttr(Attr);
    }

    FunctionType getFunctionType() {
      return clift::unwrapped_cast<FunctionType>(getType());
    }


    mlir::Type getReturnType() {
      return getFunctionType().getReturnType();
    }

    unsigned getArgCount() {
      return getFunctionType().getArgumentTypes().size();
    }


    clift::AttrDictView getArgAttrs(unsigned Index) {
      std::optional<mlir::ArrayAttr> Attrs = getArgAttrs();
      mlir::DictionaryAttr DictAttr = Attrs ?
                        mlir::cast<mlir::DictionaryAttr>((*Attrs)[Index]) :
                        mlir::DictionaryAttr::get(getContext());
      return AttrDictView(DictAttr);
    }

    clift::AttrDictView getResAttrs(unsigned Index) {
      std::optional<mlir::ArrayAttr> Attrs = getArgAttrs();
      mlir::DictionaryAttr DictAttr = Attrs ?
                        mlir::cast<mlir::DictionaryAttr>((*Attrs)[Index]) :
                        mlir::DictionaryAttr::get(getContext());
      return AttrDictView(DictAttr);
    }


    mlir::Region::BlockListType& getBlocks() {
        return getBody().getBlocks();
    }

    mlir::Block::args_iterator args_begin() {
        return getBody().front().args_begin();
    }

    mlir::Block::args_iterator args_end() {
        return getBody().front().args_end();
    }


    // Implementation of CallableOpInterface

    mlir::Region *getCallableRegion() {
        return isExternal() ? nullptr : &getBody();
    }

    /// Returns the result types of this function.
    llvm::ArrayRef<mlir::Type> getCallableResults() {
      return getResultTypes();
    }


    // Implementation of FunctionOpInterface

    llvm::ArrayRef<mlir::Type> getArgumentTypes();
    llvm::ArrayRef<mlir::Type> getResultTypes();

    mlir::Type cloneTypeWith(mlir::TypeRange inputs, mlir::TypeRange results);
  }];
}

def Clift_GlobalVariableOp : Clift_Op<"global",
                                      [Clift_GlobalOpInterface,
                                       NoRegionArguments,
                                       IsolatedFromAbove]> {
  let arguments = (ins SymbolNameAttr:$sym_name,
                       TypeAttrOf<Clift_AddressableType>:$type,
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
                             Trait interface,
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
    OpBuilder<(ins CArg<"clift::LoopOpInterface", "{}">:$OtherLoop)>
  ];
}

def Clift_MakeLabelOp : Clift_StatementOp<"make_label"> {
  let arguments = (ins DefaultValuedStrAttr<StrAttr, "">:$name,
                       DefaultValuedStrAttr<StrAttr, "">:$handle);

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
            clift::LabelType::get($_builder.getContext()));
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

def Clift_BlockStatementOp
  : Clift_StatementOp<"block", [Clift_StatementRegionOpInterface]> {

  let regions = (region Clift_StatementRegion:$block);

  let assemblyFormat = [{
    $block attr-dict-with-keyword
  }];

  let extraClassDeclaration = [{
    clift::NoFallthroughKind isIndirectlyNoFallthrough();

    unsigned getStatementRegionCount() {
      return 1;
    }

    mlir::Region &getStatementRegion(unsigned Index) {
      revng_assert(Index == 0);
      return getBlock();
    }
  }];
}

def Clift_BreakToOp : Clift_JumpStatementOp<"break_to"> {
  // The target label operand is optional: a break_to with no operand represents
  // a plain C `break` targeting the innermost enclosing loop. The operand is
  // dropped by the promote-break-continue pass when that promotion is valid.
  let arguments = (ins Optional<Clift_LabelType>:$label);

  let assemblyFormat = [{
    ($label^)? attr-dict
  }];

  let hasVerifier = 1;

  let extraClassDeclaration = [{
    [[nodiscard]] MakeLabelOp getLabelOp();
  }];
}

def Clift_ContinueToOp : Clift_JumpStatementOp<"continue_to"> {
  // The target label operand is optional: a continue_to with no operand
  // represents a plain C `continue` targeting the innermost enclosing loop. The
  // operand is dropped by the promote-break-continue pass when that promotion is
  // valid.
  let arguments = (ins Optional<Clift_LabelType>:$label);

  let assemblyFormat = [{
    ($label^)? attr-dict
  }];

  let hasVerifier = 1;

  let extraClassDeclaration = [{
    [[nodiscard]] MakeLabelOp getLabelOp();
  }];
}

def Clift_DoWhileOp : Clift_LoopStatementOp<"do_while", [NoRegionArguments]> {
  let regions = (region Clift_StatementRegion:$body,
                        Clift_ExpressionRegion:$condition);

  // The leading `` suppresses the space the printer inserts before the label
  // directive, which prints each label with its own leading space instead. This
  // keeps a label-less loop free of stray whitespace.
  let assemblyFormat = [{
    `` custom<CliftLoopLabels>($label_mask, $labels)
    `body` $body `cond` $condition attr-dict-with-keyword
  }];

  let hasVerifier = 1;

  let extraClassDeclaration = [{
    [[nodiscard]] bool isBooleanTestedExpression(mlir::Region &R) {
      return &R == &getCondition();
    }

    unsigned getStatementRegionCount() {
      return 1;
    }

    mlir::Region &getStatementRegion(unsigned Index) {
      revng_assert(Index == 0);
      return getBody();
    }

    unsigned getExpressionRegionCount() {
      return 1;
    }

    mlir::Region &getExpressionRegion(unsigned Index) {
      revng_assert(Index == 0);
      return getCondition();
    }
  }];
}

def Clift_ExpressionStatementOp
  : Clift_StatementOp<"expr", [Clift_ExpressionRegionOpInterface]> {

  let regions = (region Clift_OptionalExpressionRegion:$expression);

  let assemblyFormat = [{
    $expression attr-dict-with-keyword
  }];

  let extraClassDeclaration = [{
    [[nodiscard]] bool isDiscardedExpression(mlir::Region &R) {
      return true;
    }

    unsigned getExpressionRegionCount() {
      return 1;
    }

    mlir::Region &getExpressionRegion(unsigned Index) {
      revng_assert(Index == 0);
      return getExpression();
    }
  }];
}

def Clift_ForOp
  : Clift_LoopStatementOp<"for",
                          [DeclareOpInterfaceMethods<Clift_StatementOpInterface,
                                                     ["getBlockArgumentVariable"]>]> {

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

    unsigned getStatementRegionCount() {
      return 1;
    }

    mlir::Region &getStatementRegion(unsigned Index) {
      revng_assert(Index == 0);
      return getBody();
    }

    unsigned getExpressionRegionCount();
    mlir::Region &getExpressionRegion(unsigned Index);
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
    clift::NoFallthroughKind isIndirectlyNoFallthrough() const;

    [[nodiscard]] bool isBooleanTestedExpression(mlir::Region &R) {
      return &R == &getCondition();
    }

    unsigned getStatementRegionCount() {
      return 2;
    }

    mlir::Region &getStatementRegion(unsigned Index) {
      revng_assert(Index < 2);
      return Index == 0 ? getThen() : getElse();
    }

    unsigned getExpressionRegionCount() {
      return 1;
    }

    mlir::Region &getExpressionRegion(unsigned Index) {
      revng_assert(Index == 0);
      return getCondition();
    }
  }];
}

def Clift_LocalVariableOp
  : Clift_BasicStatementOp<"local",
                           Clift_StatementOpInterface,
                           [DeclareOpInterfaceMethods<Clift_StatementOpInterface,
                                                      ["getBlockArgumentVariable"]>]> {

  let arguments = (ins DefaultValuedStrAttr<StrAttr, "">:$name,
                       DefaultValuedStrAttr<StrAttr, "">:$handle);

  let results = (outs Clift_AnyObjectType:$result);

  let regions = (region Clift_OptionalExpressionRegion:$initializer);

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;

  let extraClassDeclaration = [{
    mlir::Type getType() {
      return getResult().getType();
    }

    unsigned getExpressionRegionCount() {
      return 1;
    }

    mlir::Region &getExpressionRegion(unsigned Index) {
      revng_assert(Index == 0);
      return getInitializer();
    }
  }];
}

def Clift_ReturnOp : Clift_StatementOp<"return", [Clift_NoFallthrough]> {

  let regions = (region Clift_OptionalExpressionRegion:$result);

  let assemblyFormat = [{
    $result attr-dict-with-keyword
  }];

  let extraClassDeclaration = [{
    unsigned getExpressionRegionCount() {
      return 1;
    }

    mlir::Region &getExpressionRegion(unsigned Index) {
      revng_assert(Index == 0);
      return getResult();
    }
  }];

  let hasVerifier = 1;
}

def Clift_RequireOp : Clift_StatementOp<"require"> {
  let arguments = (ins AnyType:$local);

  let assemblyFormat = [{
    $local attr-dict `:` type($local)
  }];

  let hasVerifier = 1;

  let extraClassDeclaration = [{
    [[nodiscard]] LocalVariableOp getLocalOp();
  }];
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
    clift::NoFallthroughKind isIndirectlyNoFallthrough() const;

    [[nodiscard]] mlir::Type getConditionType();

    [[nodiscard]] mlir::Region &getConditionRegion() {
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

    unsigned getStatementRegionCount() {
      return getOperation()->getRegions().size() - 1;
    }

    mlir::Region &getStatementRegion(unsigned Index) {
      return getOperation()->getRegions()[Index + 1];
    }

    unsigned getExpressionRegionCount() {
      return 1;
    }

    mlir::Region &getExpressionRegion(unsigned Index) {
      revng_assert(Index == 0);
      return getCondition();
    }
  }];
}

def Clift_WhileOp : Clift_LoopStatementOp<"while", [NoRegionArguments]> {
  let regions = (region Clift_ExpressionRegion:$condition,
                        Clift_StatementRegion:$body);

  // The leading `` suppresses the space the printer inserts before the label
  // directive, which prints each label with its own leading space instead. This
  // keeps a label-less loop free of stray whitespace.
  let assemblyFormat = [{
    `` custom<CliftLoopLabels>($label_mask, $labels)
    `cond` $condition `body` $body attr-dict-with-keyword
  }];

  let hasVerifier = 1;

  let extraClassDeclaration = [{
    [[nodiscard]] bool isBooleanTestedExpression(mlir::Region &R) {
      return &R == &getCondition();
    }

    unsigned getStatementRegionCount() {
      return 1;
    }

    mlir::Region &getStatementRegion(unsigned Index) {
      revng_assert(Index == 0);
      return getBody();
    }

    unsigned getExpressionRegionCount() {
      return 1;
    }

    mlir::Region &getExpressionRegion(unsigned Index) {
      revng_assert(Index == 0);
      return getCondition();
    }
  }];
}

//===----------------------------- Expressions ----------------------------===//

class Clift_BasicExpressionOp<string mnemonic,
                              Trait interface,
                              list<Trait> traits = []>
  : Clift_Op<mnemonic, !listconcat(traits, [interface,
                                            NoMemoryEffect,
                                            NoRegionArguments])>;

class Clift_ExpressionOp<string mnemonic, list<Trait> traits = []>
  : Clift_BasicExpressionOp<mnemonic, Clift_ExpressionOpInterface, traits>;

def Clift_UndefOp : Clift_ExpressionOp<"undef"> {
  let results = (outs Clift_AnyAddressableType:$result);

  let assemblyFormat = [{
    attr-dict `:` type($result)
  }];
}

def Clift_YieldOp : Clift_Op<"yield", [Clift_ExpressionOpInterface,
                                       Terminator,
                                       NoRegionArguments]> {

  let arguments = (ins Clift_AnyAddressableType:$value);

  let assemblyFormat = [{
    $value attr-dict `:` type($value)
  }];

  let extraClassDeclaration = [{
    [[nodiscard]] bool isDiscardedOperand(mlir::OpOperand &Operand);
    [[nodiscard]] bool isBooleanTestedOperand(mlir::OpOperand &Operand);
  }];
}

def Clift_ImmediateOp : Clift_ExpressionOp<"imm"> {
  let arguments = (ins I64Attr:$value);

  let results = (outs Clift_AnyNonConstType<Clift_AnyIntegerType>:$result);

  let assemblyFormat = [{
    $value attr-dict `:` type($result)
  }];
}

def Clift_StringOp : Clift_ExpressionOp<"str"> {
  let arguments = (ins StrAttr:$value);
  let results = (outs AnyType:$result);

  let assemblyFormat = [{
    $value attr-dict `:` type($result)
  }];

  let hasVerifier = 1;
}

class Clift_UnaryArithmeticOp<string mnemonic>
  : Clift_ExpressionOp<mnemonic,
                       [Clift_UnqualifiedTypeMatches<"value", "result">]> {

  let arguments = (ins Clift_AnyIntegerType:$value);
  let results = (outs Clift_AnyNonConstType<Clift_AnyIntegerType>:$result);

  let assemblyFormat = [{
    $value attr-dict `:` type($value)
  }];
}

class Clift_BinaryArithmeticOp<string mnemonic, list<Trait> traits = []>
  : Clift_ExpressionOp<mnemonic,
                       !listconcat(traits,
                                   [Clift_AllTypesMatch<["lhs", "rhs", "result"]>])> {

  let arguments = (ins Clift_AnyIntegerType:$lhs,
                       Clift_AnyIntegerType:$rhs);

  let results = (outs Clift_AnyNonConstType<Clift_AnyIntegerType>:$result);

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
  : Clift_ExpressionOp<mnemonic, traits> {

  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results = (outs Clift_AnyNonConstType<AnyType>:$result);

  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict `:` `(`
    custom<CliftPointerArithmeticOpTypes>(type($result), type($lhs), type($rhs))
    `)`
  }];

  let hasVerifier = 1;

  let extraClassDeclaration = [{
    unsigned getPointerOperandIndex() {
      return clift::impl::getPointerArithmeticPointerOperandIndex(getOperation());
    }

    mlir::Value getPointer() {
      return getOperation()->getOperand(getPointerOperandIndex());
    }

    unsigned getOffsetOperandIndex() {
      return clift::impl::getPointerArithmeticOffsetOperandIndex(getOperation());
    }

    mlir::Value getOffset() {
      return getOperation()->getOperand(getOffsetOperandIndex());
    }
  }];
}

class Clift_BinaryLogicalOp<string mnemonic>
  : Clift_ExpressionOp<mnemonic, [Clift_ReturnsBoolean]> {

  let arguments = (ins Clift_AnyScalarType:$lhs,
                       Clift_AnyScalarType:$rhs);

  let results = (outs Clift_AnyNonConstType<Clift_AnyBooleanType>:$result);

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
                       [Clift_UnqualifiedTypeMatches<"lhs", "result">]> {

  let arguments = (ins Clift_AnyIntegerType:$lhs,
                       Clift_AnyIntegerType:$rhs);

  let results = (outs Clift_AnyNonConstType<Clift_AnyIntegerType>:$result);

  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict `:` custom<CliftOpOperandTypes>(type($lhs), type($rhs))
  }];
}

class Clift_ComparisonOp<string mnemonic>
  : Clift_ExpressionOp<mnemonic,
                       [Commutative,
                        Clift_ReturnsBoolean,
                        Clift_AllTypesMatch<["lhs", "rhs"]>]> {

  let arguments = (ins Clift_AnyScalarType:$lhs,
                       Clift_AnyScalarType:$rhs);

  let results = (outs Clift_AnyNonConstType<Clift_AnyBooleanType>:$result);

  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict `:`
    custom<CliftOpOperandTypes>(type($lhs), type($rhs)) `->` type($result)
  }];
}

class Clift_UnaryIntegerMutationOp<string mnemonic>
  : Clift_ExpressionOp<mnemonic,
                       [AllTypesMatch<["value", "result"]>,
                        Clift_LValueOperand<"value">]> {

  let arguments = (ins Clift_AnyModifiableType<Clift_AnyIntegerType>:$value);
  let results = (outs Clift_AnyIntegerType:$result);

  let assemblyFormat = [{
    $value attr-dict `:` type($value)
  }];

  let extraClassDeclaration = [{
    bool isLvalueExpression() {
      return true;
    }
  }];
}

def Clift_NegOp : Clift_UnaryArithmeticOp<"neg">;
def Clift_AddOp : Clift_BinaryArithmeticOp<"add", [Commutative]>;
def Clift_SubOp : Clift_BinaryArithmeticOp<"sub">;
def Clift_MulOp : Clift_BinaryArithmeticOp<"mul", [Commutative]>;
def Clift_SDivOp : Clift_BinaryArithmeticOp<"sdiv">;
def Clift_UDivOp : Clift_BinaryArithmeticOp<"udiv">;
def Clift_SRemOp : Clift_BinaryArithmeticOp<"srem">;
def Clift_URemOp : Clift_BinaryArithmeticOp<"urem">;

def Clift_PtrAddOp : Clift_PointerArithmeticOp<"ptr_add", [Commutative]>;
def Clift_PtrSubOp : Clift_PointerArithmeticOp<"ptr_sub">;

def Clift_PtrDiffOp : Clift_ExpressionOp<"ptr_diff"> {
  let arguments = (ins Clift_AnyPointerType:$lhs,
                       Clift_AnyPointerType:$rhs);

  let results = (outs Clift_AnyNonConstType<AnyType>:$result);

  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict `:`
    custom<CliftOpOperandTypes>(type($lhs), type($rhs)) `->` type($result)
  }];

  let hasVerifier = 1;
}

def Clift_LogicalNotOp
  : Clift_ExpressionOp<"not", [Clift_ReturnsBoolean]> {

  let arguments = (ins Clift_AnyIntegerType:$value);
  let results = (outs Clift_AnyNonConstType<Clift_AnyBooleanType>:$result);

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

def Clift_ShlOp : Clift_ShiftOp<"shl">;
def Clift_ShrOp : Clift_ShiftOp<"shr">;
def Clift_SarOp : Clift_ShiftOp<"sar">;

def Clift_CmpEqOp : Clift_ComparisonOp<"eq">;
def Clift_CmpNeOp : Clift_ComparisonOp<"ne">;

def Clift_SCmpLtOp : Clift_ComparisonOp<"slt">;
def Clift_UCmpLtOp : Clift_ComparisonOp<"ult">;
def Clift_SCmpGtOp : Clift_ComparisonOp<"sgt">;
def Clift_UCmpGtOp : Clift_ComparisonOp<"ugt">;
def Clift_SCmpLeOp : Clift_ComparisonOp<"sle">;
def Clift_UCmpLeOp : Clift_ComparisonOp<"ule">;
def Clift_SCmpGeOp : Clift_ComparisonOp<"sge">;
def Clift_UCmpGeOp : Clift_ComparisonOp<"uge">;

def Clift_IncrementOp : Clift_UnaryIntegerMutationOp<"inc">;
def Clift_DecrementOp : Clift_UnaryIntegerMutationOp<"dec">;

def Clift_PostIncrementOp : Clift_UnaryIntegerMutationOp<"post_inc">;
def Clift_PostDecrementOp : Clift_UnaryIntegerMutationOp<"post_dec">;

class Clift_CastExpressionOp<string mnemonic, list<Trait> traits = []>
  : Clift_BasicExpressionOp<mnemonic, Clift_CastOpInterface, traits>;

def Clift_DecayOp : Clift_CastExpressionOp<"decay"> {
  let arguments = (ins AnyType:$value);

  let results = (outs Clift_AnyNonConstType<AnyType>:$result);

  let assemblyFormat = [{
    $value attr-dict `:` type($value) `->` type($result)
  }];

  let hasVerifier = 1;
}

def Clift_BitCastOp
  : Clift_CastExpressionOp<"bitcast",
                           [Clift_AllSizesMatch<["value", "result"]>]> {

  let arguments = (ins Clift_AnyValueType:$value);
  let results = (outs Clift_AnyNonConstType<Clift_AnyValueType>:$result);

  let assemblyFormat = [{
    $value attr-dict `:` type($value) `->` type($result)
  }];
}

def Clift_ZeroExtendOp
  : Clift_CastExpressionOp<"zext",
                           [Clift_ObjectSizesAreOrdered<"value", "result">]> {

  let arguments = (ins Clift_AnyIntegerType:$value);

  let results = (outs Clift_AnyNonConstType<Clift_AnyIntegerType>:$result);

  let assemblyFormat = [{
    $value attr-dict `:` type($value) `->` type($result)
  }];
}

def Clift_SignExtendOp
  : Clift_CastExpressionOp<"sext",
                           [Clift_ObjectSizesAreOrdered<"value", "result">]> {

  let arguments = (ins Clift_AnyIntegerType:$value);

  let results = (outs Clift_AnyNonConstType<Clift_AnyIntegerType>:$result);

  let assemblyFormat = [{
    $value attr-dict `:` type($value) `->` type($result)
  }];
}

def Clift_TruncateOp
  : Clift_CastExpressionOp<"truncate",
                           [Clift_ObjectSizesAreOrdered<"result", "value">]> {

  let arguments = (ins Clift_AnyIntegerType:$value);

  let results = (outs Clift_AnyNonConstType<Clift_AnyIntegerType>:$result);

  let assemblyFormat = [{
    $value attr-dict `:` type($value) `->` type($result)
  }];
}

def Clift_PtrResizeOp
  : Clift_CastExpressionOp<"ptr_resize",
                           [Clift_AllPointeeTypesMatch<["value", "result"]>]> {

  let arguments = (ins Clift_AnyPointerType:$value);
  let results = (outs Clift_AnyNonConstType<Clift_AnyPointerType>:$result);

  let assemblyFormat = [{
    $value attr-dict `:` type($value) `->` type($result)
  }];

  let hasVerifier = 1;
}

def Clift_AddressofOp
  : Clift_ExpressionOp<"addressof",
                       [Clift_PointeeTypeMatches<"result", "object">,
                        Clift_LValueOperand<"object">]> {

  let arguments = (ins AnyType:$object);
  let results = (outs Clift_AnyNonConstType<Clift_AnyPointerType>:$result);

  let assemblyFormat = [{
    $object attr-dict `:` type($result)
  }];
}

def Clift_IndirectionOp
  : Clift_ExpressionOp<"indirection",
                       [Clift_PointeeTypeMatches<"pointer", "result">]> {

  let arguments = (ins Clift_AnyPointerType:$pointer);
  let results = (outs Clift_AnyNonVoidType:$result);

  let assemblyFormat = [{
    $pointer attr-dict `:` type($pointer)
  }];

  let extraClassDeclaration = [{
    bool isLvalueExpression() {
      return true;
    }
  }];
}

def Clift_AssignOp
  : Clift_ExpressionOp<"assign",
                       [AllTypesMatch<["lhs", "result"]>,
                        Clift_AllTypesMatch<["lhs", "rhs"]>,
                        Clift_LValueOperand<"lhs">]> {

  let arguments = (ins Clift_AnyModifiableType<Clift_AnyValueType>:$lhs,
                       AnyType:$rhs);

  let results = (outs AnyType:$result);

  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict `:`
    custom<CliftOpOperandTypes>(type($lhs), type($rhs))
  }];

  let extraClassDeclaration = [{
    bool isLvalueExpression() {
      return true;
    }
  }];
}

def Clift_AccessOp : Clift_ExpressionOp<"access"> {
  let arguments = (ins AnyType:$value,
                       UnitAttr:$indirect,
                       I64Attr:$member_index);

  let results = (outs AnyType:$result);

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
                       [Clift_PointeeTypeMatches<"pointer", "result">]> {
  let arguments = (ins AnyType:$pointer,
                       Clift_AnyIntegerType:$index);

  let results = (outs AnyType:$result);

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

  let arguments = (ins Clift_AnyAddressableType:$lhs,
                       Clift_AnyAddressableType:$rhs);

  let results = (outs AnyType:$result);

  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs)
  }];

  let extraClassDeclaration = [{
    bool isLvalueExpression() {
      return clift::isLvalueExpression(getRhs());
    }

    [[nodiscard]] bool isDiscardedOperand(mlir::OpOperand &Operand) {
      return &Operand == &getOperation()->getOpOperand(0);
    }
  }];
}

def Clift_UseOp
  : Clift_ExpressionOp<"use",
                       [DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {

  let arguments = (ins FlatSymbolRefAttr:$symbol_name);
  let results = (outs Clift_AnyAddressableType:$result);

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
  let arguments = (ins AnyType:$function,
                       Variadic<AnyType>:$arguments);

  let results = (outs Clift_AnyNonConstType<AnyType>:$result);

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;

  let extraClassDeclaration = [{
    FunctionType getFunctionType();
  }];
}

def Clift_TernaryOp
  : Clift_ExpressionOp<"ternary",
                       [Clift_AllTypesMatch<["lhs", "rhs"]>,
                        Clift_UnqualifiedTypeMatches<"lhs", "result">]> {

  let arguments = (ins Clift_AnyScalarType:$condition,
                       AnyType:$lhs,
                       AnyType:$rhs);

  let results = (outs Clift_AnyNonConstType<Clift_AnyAddressableType>:$result);

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

def Clift_AggregateOp : Clift_ExpressionOp<"aggregate"> {
  let arguments = (ins Variadic<AnyType>:$initializers);
  let results = (outs Clift_AnyNonConstType<AnyType>:$result);

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

#endif
