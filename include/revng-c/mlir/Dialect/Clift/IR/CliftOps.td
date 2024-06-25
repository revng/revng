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

def Clift_LoopOp : Clift_Op<"loop", [Terminator]> {
  let regions = (region AnyRegion:$body);
  let successors = (successor
    VariadicSuccessor<AnySuccessor>:$LoopDestinations);

  let assemblyFormat = [{
    $body $LoopDestinations attr-dict
  }];

  let hasRegionVerifier = 1;
}

def Clift_ContinueOp : Clift_Op<"continue", [HasParent<"LoopOp">, Terminator]> {}

def Clift_LabelResource : Resource<"LabelResource">;

def Clift_AssignLabelOp : Clift_Op<"assign_label"> {
  let arguments = (ins Arg<Clift_LabelType, "", [MemFree]>:$label);

  let assemblyFormat = [{
    $label type($label) attr-dict
  }];

  let hasCanonicalizeMethod = 1;
}

def Clift_GoToOp : Clift_Op<"goto", [Terminator]> {
  let arguments = (ins Arg<Clift_LabelType, "", [MemRead]>:$label);

  let assemblyFormat = [{
    $label type($label) attr-dict
  }];
}

def Clift_MakeLabelOp : Clift_Op<"make_label", [Clift_OneUseOfType<"AssignLabelOp">] > {
  let results = (outs Res<Clift_LabelType, "", [MemAlloc, MemWrite] >:$result);

  let assemblyFormat = [{
    type($result) attr-dict
  }];
  let hasVerifier = 1;

  let builders = [
     OpBuilder<(ins), [{
       build($_builder, $_state, mlir::clift::LabelType::get($_builder.getContext()));
   }]>
   ];
}

def Clift_ConstantOp: Clift_Op<"constant", [Pure]> {
  let arguments = (ins AnyAttrOf<[I64Attr, F64Attr, BoolAttr]>:$value);

  let results = (outs Clift_PrimitiveType:$result);

  let assemblyFormat = [{
    $value type($result) attr-dict
  }];

   let builders = [
        OpBuilder<(ins "mlir::Type":$type, "double":$value), [{
            build($_builder, $_state, type, $_builder.getF64FloatAttr(value));
    }]>,OpBuilder<(ins "mlir::Type":$type, "bool":$value), [{
            build($_builder, $_state, type, $_builder.getBoolAttr(value));
    }]>,OpBuilder<(ins "mlir::Type":$type, "std::int64_t":$value), [{
            build($_builder, $_state, type, $_builder.getI64IntegerAttr(value));
    }]>
    ];
}

#endif
