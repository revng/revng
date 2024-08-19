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

class Clift_Op<string mnemonic, list<Trait> traits = []> :
    Op<Clift_Dialect, mnemonic, traits>;

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
                               NoTerminator,
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
                                 IsolatedFromAbove,
                                 NoTerminator]> {
  let arguments = (ins StrAttr:$sym_name,
                       TypeAttrOf<FunctionType>:$function_type,
                       OptionalAttr<DictArrayAttr>:$arg_attrs,
                       OptionalAttr<DictArrayAttr>:$res_attrs);

  let description = [{
    expression.
  }];

  let regions = (region Clift_StatementRegion:$body);

  let assemblyFormat = [{
    $sym_name $function_type $body attr-dict
  }];

  code extraClassDeclaration = [{
    Region *getCallableRegion() {
        return isExternal() ? nullptr : &getBody();
    }

    mlir::Region::BlockListType& getBlocks() {
        return getBody().getBlocks();
    }

    /// Returns the result types of this function.
    /// Part of CallableOpInterface.
    ArrayRef<Type> getCallableResults() {
        return getFunctionType().getResults();
    }

    /// Returns the argument types of this function.
    /// Part of FunctionOpInterface.
    ArrayRef<Type> getArgumentTypes() {
        return getFunctionType().getInputs();
    }

    /// Returns the result types of this function.
    /// Part of FunctionOpInterface.
    ArrayRef<Type> getResultTypes() {
        return getFunctionType().getResults();
    }

    /// Part of FunctionOpInterface.
    bool isExternal() {
        return getBody().empty();
    }

    Block::args_iterator args_begin() {
        return getBody().front().args_begin();
    }

    Block::args_iterator args_end() {
        return getBody().front().args_end();
    }
  }];
}

def Clift_GlobalVariableOp : Clift_Op<"global",
                                      [Symbol,
                                       IsolatedFromAbove,
                                       NoTerminator]> {
  let arguments = (ins StrAttr:$sym_name,
                       TypeAttrOf<Clift_ValueType>:$type);

  let description = [{
    global variable
  }];

  let regions = (region Clift_OptionalExpressionRegion:$initializer);

  let assemblyFormat = [{
    $type $sym_name (`=` $initializer^)? attr-dict
  }];
}

def Clift_UndefOp: Clift_Op<"undef"> {
  let results = (outs Clift_AnyValueType:$result);

  let assemblyFormat = [{
    type($result) attr-dict
  }];
}

//===----------------------------- Statements -----------------------------===//

def Clift_MakeLabelOp : Clift_Op<"make_label",
                                 [Clift_StatementOpInterface,
                                  Clift_OneUseOfType<"mlir::clift::AssignLabelOp">] > {
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

def Clift_AssignLabelOp : Clift_Op<"assign_label",
                                   [Clift_StatementOpInterface]> {
  let arguments = (ins Clift_LabelType:$label);

  let assemblyFormat = [{
    $label type($label) attr-dict
  }];
}

def Clift_GoToOp : Clift_Op<"goto",
                            [Clift_StatementOpInterface,
                             Clift_NoFallthrough]> {
  let arguments = (ins Clift_LabelType:$label);

  let assemblyFormat = [{
    $label type($label) attr-dict
  }];
}

def Clift_IfOp : Clift_Op<"if",
                          [Clift_StatementOpInterface,
                           NoTerminator]> {
  let regions = (region Clift_ExpressionRegion:$condition,
                        Clift_StatementRegion:$then,
                        Clift_StatementRegion:$else);

  let assemblyFormat = [{
    $condition $then (`else` $else^)? attr-dict
  }];
}

def Clift_LocalVariableOp : Clift_Op<"local",
                                     [Clift_StatementOpInterface,
                                      NoTerminator]> {
  let arguments = (ins StrAttr:$sym_name,
                       TypeAttrOf<Clift_ValueType>:$type);

  let regions = (region Clift_OptionalExpressionRegion:$initializer);

  let assemblyFormat = [{
    $type $sym_name (`=` $initializer^)? attr-dict
  }];
}

def Clift_ReturnOp : Clift_Op<"return",
                              [Clift_StatementOpInterface,
                               Clift_NoFallthrough,
                               NoTerminator]> {
  let regions = (region Clift_OptionalExpressionRegion:$result);

  let assemblyFormat = [{
    $result attr-dict
  }];
}

def Clift_SwitchOp : Clift_Op<"switch",
                              [Clift_StatementOpInterface,
                               NoTerminator]> {
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

def Clift_SwitchBreakOp : Clift_Op<"switch_break",
                                   [Clift_StatementOpInterface,
                                    Clift_NoFallthrough]> {
  let assemblyFormat = [{
    attr-dict
  }];
}

#endif
