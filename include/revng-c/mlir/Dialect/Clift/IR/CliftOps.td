//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#ifndef MLIR_CLIFT_OPS
#define MLIR_CLIFT_OPS

include "mlir/Interfaces/CallInterfaces.td"
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

def Clift_AnyValueType : Type<CPred<"llvm::isa<mlir::clift::ValueType>($_self)">,
                              "Clift value type",
                              "mlir::clift::ValueType">;

def Clift_FunctionType : Type<CPred<"mlir::clift::impl::verifyFunctionType($_self)">,
                              "Clift function type",
                              "mlir::clift::ValueType">;

//===---------------------------- Region types ----------------------------===//

def Clift_StatementRegion : Region<CPred<"::mlir::clift::impl::verifyStatementRegion($_self)">,
                                   "Region containing statements">;

def Clift_ExpressionRegion : Region<CPred<"::mlir::clift::impl::verifyExpressionRegion($_self, true)">,
                                    "Region representing an expression">;

def Clift_OptionalExpressionRegion : Region<CPred<"::mlir::clift::impl::verifyExpressionRegion($_self, false)">,
                                            "Optional region representing an expression">;

//===------------------------ Top level operations ------------------------===//

def Clift_ModuleOp : Clift_Op<"module",
                              [SymbolTable,
                               HasOnlyGraphRegion,
                               NoRegionArguments,
                               SingleBlock,
                               RegionKindInterface]> {
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
                       TypeAttrOf<Clift_FunctionType>:$function_type,
                       DefaultValuedStrAttr<StrAttr, "">:$unique_handle);

  let description = [{
    expression.
  }];

  let regions = (region Clift_StatementRegion:$body);

  let assemblyFormat = [{
    $sym_name $function_type $body attr-dict
  }];

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

    mlir::ArrayAttr getArgAttrsAttr();
    mlir::ArrayAttr getResAttrsAttr();

    void setArgAttrsAttr(mlir::ArrayAttr attrs);
    void setResAttrsAttr(mlir::ArrayAttr attrs);

    mlir::Attribute removeArgAttrsAttr();
    mlir::Attribute removeResAttrsAttr();

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

def Clift_UndefOp : Clift_Op<"undef",
                             [Clift_ExpressionOpInterface]> {
  let results = (outs Clift_AnyValueType:$result);

  let assemblyFormat = [{
    `:` type($result) attr-dict
  }];
}

def Clift_YieldOp : Clift_Op<"yield",
                             [Clift_ExpressionOpInterface,
                              Terminator]> {
  let arguments = (ins Clift_AnyValueType:$value);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $value `:` type($value) attr-dict
  }];
}

#endif
