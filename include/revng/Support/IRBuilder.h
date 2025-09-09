#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/IRBuilder.h"

#include "revng/Support/Debug.h"

// Placing the wrapper in the llvm namespace is necessary to avoid manual
// prefixing of every single llvm type with `llvm::` in the auto-generated
// part of the class definition.
//
// Only one extra name is added to `llvm` namespace because of this is
// `RevngIRBuilderWrapper`. Please avoid adding any more to minimize
// the possibility of name collisions.
namespace llvm {

/// This is a wrapper over llvm alternative that force-sets a debug location
/// even when it's the insertion point is a basic block.
///
/// On top of that, it asserts that *every* created instruction is created
/// with valid debug information attached.
class RevngIRBuilderWrapper {

private:
  // NOLINTNEXTLINE
  llvm::IRBuilder<> Underlying;

public:
  //
  // Add extra debug location related logic in constructors.
  //

  RevngIRBuilderWrapper(llvm::BasicBlock *BB, const llvm::DebugLoc &DL) :
    Underlying(BB) {

    if (DL)
      Underlying.SetCurrentDebugLocation(DL);
  }

  /// This overload should be avoided in favor of the one that explicitly
  /// provides a debug location.
  explicit RevngIRBuilderWrapper(llvm::BasicBlock *BB) :
    RevngIRBuilderWrapper(BB,
                          BB->getTerminator() ?
                            BB->getTerminator()->getDebugLoc() :
                            llvm::DebugLoc{}) {}

  explicit RevngIRBuilderWrapper(llvm::Instruction *I) : Underlying(I) {}

  RevngIRBuilderWrapper(llvm::BasicBlock *BB, llvm::BasicBlock::iterator I) :
    Underlying(BB, I) {}

  explicit RevngIRBuilderWrapper(llvm::LLVMContext &C) : Underlying(C) {}

public:
  //
  // Same with explicit invocations of `SetInsertPoint`.
  //

  void SetInsertPoint(llvm::BasicBlock *BB, const llvm::DebugLoc &DL) {
    Underlying.SetInsertPoint(BB);

    if (DL)
      Underlying.SetCurrentDebugLocation(DL);
  }
  void SetInsertPoint(llvm::BasicBlock *BB) {
    SetInsertPoint(BB,
                   BB->getTerminator() ? BB->getTerminator()->getDebugLoc() :
                                         llvm::DebugLoc{});
  }
  void SetInsertPoint(llvm::Instruction *I) { Underlying.SetInsertPoint(I); }
  void SetInsertPoint(llvm::BasicBlock *BB, llvm::BasicBlock::iterator I) {
    Underlying.SetInsertPoint(BB, I);
  }
  void SetInsertPointPastAllocas(llvm::Function *F) {
    Underlying.SetInsertPointPastAllocas(F);

    llvm::DebugLoc DebugLocation = Underlying.GetInsertPoint()->getDebugLoc();
    Underlying.SetCurrentDebugLocation(DebugLocation);
  }

public:
  /// This is a way to disable checks for a specific instance of
  /// `revng::IRBuilder`. Please use with care!
  ///
  /// Practically, this should only appear in one of two cases:
  /// - in `revngLift` before debug information has been attached.
  /// - in the old pipeline as a temporary workaround to avoid having to
  ///   fix selected issues.
  bool ChecksEnabled = true;

private:
  void checkImpl() const {
    if (ChecksEnabled) {
      // TODO: adopt `isDebugLocationInvalid` here once it's available.
      llvm::DebugLoc CurrentLocation = Underlying.getCurrentDebugLocation();
      revng_assert(CurrentLocation);
      revng_assert(CurrentLocation->getScope());
      revng_assert(not CurrentLocation->getScope()->getName().empty());
    }
  }

public:
  // Since one if `Insert` members is a template, instrument it manually.
  template<typename InstTy>
  InstTy *Insert(InstTy *I, const llvm::Twine &Name = "") const {
    checkImpl();
    return Underlying.Insert(I, Name);
  }

  // Provide its overloads too, to avoid having to make auto-generated
  // methods `const`-aware.
  llvm::Constant *Insert(llvm::Constant *C,
                         const llvm::Twine &Name = "") const {
    return Underlying.Insert(C, Name);
  }
  llvm::Value *Insert(llvm::Value *C, const llvm::Twine &Name = "") const {
    return Underlying.Insert(C, Name);
  }

  // Provided to resolve 'CreateIntCast(Ptr, Ptr, "...")', giving a
  // compile time error, instead of converting the string to bool for the
  // isSigned parameter.
  Value *CreateIntCast(Value *, Type *, const char *) = delete;

public:
  // This class is provided because there are members below referring to it.
  // NOLINTNEXTLINE
  using InsertPoint = llvm::IRBuilder<>::InsertPoint;

  // Guards are omitted because they are unused, but they can be added.
  // class InsertPointGuard;
  // class FastMathFlagGuard;
  // class OperandBundlesGuard;

public:
  // Everything else is generated automatically.
  //
  // The following macro allows to heavily reduce the among of boilerplate
  // code needed to wrap the original builder.
  //
  // Its arguments are:
  // - `METHOD_NAME` - the name of the method this replaces
  //   (for example, `CreateGlobalString`)
  // - `RETURN_VALUE` - the return value of the method
  //   (for example, `llvm::GlobalVariable *`)
  // - `ARGUMENT_DEFINITIONS` - the list of the method's arguments how they
  //   appear in the declaration.
  //   (for example,
  //    ```
  //    ARGS(llvm::StringRef Str,
  //         const llvm::Twine &Name = "",
  //         unsigned AddressSpace = 0,
  //         llvm::Module *M = nullptr)
  //    ```)
  // - `ARGUMENT_VALUES` - the list of the method's argument names
  //   (when passing them to `Underlying`).
  //   (for example, `ARGS(Str, Name, AddressSpace, M)`)
  //
  // Note the usage of the no-op `ARGS` macro that allows commas to be carried
  // over.
  //
  // ----------------------------
  //
  // Note that these macros are `undef`ed at the bottom of this header.

#define ARGS(...) __VA_ARGS__
#define EMBED_THE_CHECK(METHOD_NAME,                \
                        RETURN_VALUE,               \
                        ARGUMENT_DEFINITIONS,       \
                        ARGUMENT_VALUES)            \
  RETURN_VALUE METHOD_NAME(ARGUMENT_DEFINITIONS) {  \
    checkImpl();                                    \
    return Underlying.METHOD_NAME(ARGUMENT_VALUES); \
  }
#define SKIP_THE_CHECK(METHOD_NAME,                 \
                       RETURN_VALUE,                \
                       ARGUMENT_DEFINITIONS,        \
                       ARGUMENT_VALUES)             \
  RETURN_VALUE METHOD_NAME(ARGUMENT_DEFINITIONS) {  \
    return Underlying.METHOD_NAME(ARGUMENT_VALUES); \
  }

public:
  // The remainder of this class (until it's closing `};`) is generated using
  // the script below. When rebasing llvm, please rerun it.
  //
  // NOTE: it also requires minimal manual post-processing:
  //
  // - Functions that are already provided above (like `SetInsertPoint` and
  //   `Insert`) need to be removed.
  //
  // - `::` in default values confuses `clang.cindex.Index`, so stuff like
  //   like `std::nullopt` and `SyncScope::System` needs to be restored manually
  //   (grep for ` =,` and ` =)`).
  //
  // - the function marked as `= delete` in the original builder
  //   (`CreateIntCast`) needs to be manually removed.
  //
  // - replace non-`Create` function macro from `EMBED_THE_CHECK` to
  //   `SKIP_THE_CHECK` as those are not supposed to add any instructions.
  //
  // The script:
  // ```python
  // #!/usr/bin/env python3
  //
  // import sys
  // from clang.cindex import CursorKind, Index
  //
  // def main(args):
  //   if len(args) != 1:
  //     raise Exception("Please provide input file name")
  //
  //   index = Index.create()
  //   parsed = index.parse(args[0], args=["-std=c++20"])
  //
  //   for current in parsed.cursor.walk_preorder():
  //     if current.kind == CursorKind.FUNCTION_DECL:
  //       a_defs = []
  //       a_names = []
  //       for argument in current.get_arguments():
  //         a_tokens = list(argument.get_tokens())
  //         a_defs.append(" ".join(t.spelling for t in a_tokens))
  //         if argument.spelling:
  //           a_names.append(argument.spelling)
  //         else:
  //           a_defs[-1] = a_defs[-1] + f" argument_{len(a_names)}"
  //           a_names.append(f"argument_{len(a_names)}")
  //
  //       tokens = [t.spelling for t in current.get_tokens()]
  //       name_index = tokens.index(current.spelling)
  //       print(f"EMBED_THE_CHECK({current.spelling}, "
  //             f"{" ".join(tokens[:name_index])}, "
  //             f"ARGS({", ".join(a_defs)}), "
  //             f"ARGS({", ".join(a_names)}));")
  //
  // if __name__ == "__main__":
  //   sys.exit(main(sys.argv[1:]))
  // ```

  SKIP_THE_CHECK(ClearInsertionPoint, void, ARGS(), ARGS());
  SKIP_THE_CHECK(GetInsertBlock, BasicBlock *, ARGS(), ARGS());
  SKIP_THE_CHECK(GetInsertPoint, BasicBlock ::iterator, ARGS(), ARGS());
  SKIP_THE_CHECK(getContext, LLVMContext &, ARGS(), ARGS());
  SKIP_THE_CHECK(SetCurrentDebugLocation, void, ARGS(DebugLoc L), ARGS(L));
  SKIP_THE_CHECK(CollectMetadataToCopy,
                 void,
                 ARGS(Instruction *Src, ArrayRef<unsigned> MetadataKinds),
                 ARGS(Src, MetadataKinds));
  SKIP_THE_CHECK(getCurrentDebugLocation, DebugLoc, ARGS(), ARGS());
  SKIP_THE_CHECK(SetInstDebugLocation, void, ARGS(Instruction *I), ARGS(I));
  SKIP_THE_CHECK(AddMetadataToInst, void, ARGS(Instruction *I), ARGS(I));
  SKIP_THE_CHECK(getCurrentFunctionReturnType, Type *, ARGS(), ARGS());
  SKIP_THE_CHECK(saveIP, InsertPoint, ARGS(), ARGS());
  SKIP_THE_CHECK(saveAndClearIP, InsertPoint, ARGS(), ARGS());
  SKIP_THE_CHECK(restoreIP, void, ARGS(InsertPoint IP), ARGS(IP));
  SKIP_THE_CHECK(getDefaultFPMathTag, MDNode *, ARGS(), ARGS());
  SKIP_THE_CHECK(getFastMathFlags, FastMathFlags &, ARGS(), ARGS());
  SKIP_THE_CHECK(clearFastMathFlags, void, ARGS(), ARGS());
  SKIP_THE_CHECK(setDefaultFPMathTag,
                 void,
                 ARGS(MDNode *FPMathTag),
                 ARGS(FPMathTag));
  SKIP_THE_CHECK(setFastMathFlags,
                 void,
                 ARGS(FastMathFlags NewFMF),
                 ARGS(NewFMF));
  SKIP_THE_CHECK(setIsFPConstrained, void, ARGS(bool IsCon), ARGS(IsCon));
  SKIP_THE_CHECK(getIsFPConstrained, bool, ARGS(), ARGS());
  SKIP_THE_CHECK(setDefaultConstrainedExcept,
                 void,
                 ARGS(fp ::ExceptionBehavior NewExcept),
                 ARGS(NewExcept));
  SKIP_THE_CHECK(setDefaultConstrainedRounding,
                 void,
                 ARGS(RoundingMode NewRounding),
                 ARGS(NewRounding));
  SKIP_THE_CHECK(getDefaultConstrainedExcept,
                 fp ::ExceptionBehavior,
                 ARGS(),
                 ARGS());
  SKIP_THE_CHECK(getDefaultConstrainedRounding, RoundingMode, ARGS(), ARGS());
  SKIP_THE_CHECK(setConstrainedFPFunctionAttr, void, ARGS(), ARGS());
  SKIP_THE_CHECK(setConstrainedFPCallAttr, void, ARGS(CallBase *I), ARGS(I));
  SKIP_THE_CHECK(setDefaultOperandBundles,
                 void,
                 ARGS(ArrayRef<OperandBundleDef> OpBundles),
                 ARGS(OpBundles));
  SKIP_THE_CHECK(getInt1, ConstantInt *, ARGS(bool V), ARGS(V));
  SKIP_THE_CHECK(getTrue, ConstantInt *, ARGS(), ARGS());
  SKIP_THE_CHECK(getFalse, ConstantInt *, ARGS(), ARGS());
  SKIP_THE_CHECK(getInt8, ConstantInt *, ARGS(uint8_t C), ARGS(C));
  SKIP_THE_CHECK(getInt16, ConstantInt *, ARGS(uint16_t C), ARGS(C));
  SKIP_THE_CHECK(getInt32, ConstantInt *, ARGS(uint32_t C), ARGS(C));
  SKIP_THE_CHECK(getInt64, ConstantInt *, ARGS(uint64_t C), ARGS(C));
  SKIP_THE_CHECK(getIntN,
                 ConstantInt *,
                 ARGS(unsigned N, uint64_t C),
                 ARGS(N, C));
  SKIP_THE_CHECK(getInt, ConstantInt *, ARGS(const APInt &AI), ARGS(AI));
  SKIP_THE_CHECK(getInt1Ty, IntegerType *, ARGS(), ARGS());
  SKIP_THE_CHECK(getInt8Ty, IntegerType *, ARGS(), ARGS());
  SKIP_THE_CHECK(getInt16Ty, IntegerType *, ARGS(), ARGS());
  SKIP_THE_CHECK(getInt32Ty, IntegerType *, ARGS(), ARGS());
  SKIP_THE_CHECK(getInt64Ty, IntegerType *, ARGS(), ARGS());
  SKIP_THE_CHECK(getInt128Ty, IntegerType *, ARGS(), ARGS());
  SKIP_THE_CHECK(getIntNTy, IntegerType *, ARGS(unsigned N), ARGS(N));
  SKIP_THE_CHECK(getHalfTy, Type *, ARGS(), ARGS());
  SKIP_THE_CHECK(getBFloatTy, Type *, ARGS(), ARGS());
  SKIP_THE_CHECK(getFloatTy, Type *, ARGS(), ARGS());
  SKIP_THE_CHECK(getDoubleTy, Type *, ARGS(), ARGS());
  SKIP_THE_CHECK(getVoidTy, Type *, ARGS(), ARGS());
  SKIP_THE_CHECK(getPtrTy,
                 PointerType *,
                 ARGS(unsigned AddrSpace = 0),
                 ARGS(AddrSpace));
  SKIP_THE_CHECK(getInt8PtrTy,
                 PointerType *,
                 ARGS(unsigned AddrSpace = 0),
                 ARGS(AddrSpace));
  SKIP_THE_CHECK(getIntPtrTy,
                 IntegerType *,
                 ARGS(const DataLayout &DL, unsigned AddrSpace = 0),
                 ARGS(DL, AddrSpace));

  EMBED_THE_CHECK(CreateGlobalString,
                  GlobalVariable *,
                  ARGS(StringRef Str,
                       const Twine &Name = "",
                       unsigned AddressSpace = 0,
                       Module *M = nullptr),
                  ARGS(Str, Name, AddressSpace, M));
  EMBED_THE_CHECK(CreateMemSet,
                  CallInst *,
                  ARGS(Value *Ptr,
                       Value *Val,
                       uint64_t Size,
                       MaybeAlign Align,
                       bool isVolatile = false,
                       MDNode *TBAATag = nullptr,
                       MDNode *ScopeTag = nullptr,
                       MDNode *NoAliasTag = nullptr),
                  ARGS(Ptr,
                       Val,
                       Size,
                       Align,
                       isVolatile,
                       TBAATag,
                       ScopeTag,
                       NoAliasTag));
  EMBED_THE_CHECK(CreateMemSet,
                  CallInst *,
                  ARGS(Value *Ptr,
                       Value *Val,
                       Value *Size,
                       MaybeAlign Align,
                       bool isVolatile = false,
                       MDNode *TBAATag = nullptr,
                       MDNode *ScopeTag = nullptr,
                       MDNode *NoAliasTag = nullptr),
                  ARGS(Ptr,
                       Val,
                       Size,
                       Align,
                       isVolatile,
                       TBAATag,
                       ScopeTag,
                       NoAliasTag));
  EMBED_THE_CHECK(CreateMemSetInline,
                  CallInst *,
                  ARGS(Value *Dst,
                       MaybeAlign DstAlign,
                       Value *Val,
                       Value *Size,
                       bool IsVolatile = false,
                       MDNode *TBAATag = nullptr,
                       MDNode *ScopeTag = nullptr,
                       MDNode *NoAliasTag = nullptr),
                  ARGS(Dst,
                       DstAlign,
                       Val,
                       Size,
                       IsVolatile,
                       TBAATag,
                       ScopeTag,
                       NoAliasTag));
  EMBED_THE_CHECK(CreateElementUnorderedAtomicMemSet,
                  CallInst *,
                  ARGS(Value *Ptr,
                       Value *Val,
                       uint64_t Size,
                       Align Alignment,
                       uint32_t ElementSize,
                       MDNode *TBAATag = nullptr,
                       MDNode *ScopeTag = nullptr,
                       MDNode *NoAliasTag = nullptr),
                  ARGS(Ptr,
                       Val,
                       Size,
                       Alignment,
                       ElementSize,
                       TBAATag,
                       ScopeTag,
                       NoAliasTag));
  EMBED_THE_CHECK(CreateElementUnorderedAtomicMemSet,
                  CallInst *,
                  ARGS(Value *Ptr,
                       Value *Val,
                       Value *Size,
                       Align Alignment,
                       uint32_t ElementSize,
                       MDNode *TBAATag = nullptr,
                       MDNode *ScopeTag = nullptr,
                       MDNode *NoAliasTag = nullptr),
                  ARGS(Ptr,
                       Val,
                       Size,
                       Alignment,
                       ElementSize,
                       TBAATag,
                       ScopeTag,
                       NoAliasTag));
  EMBED_THE_CHECK(CreateMemCpy,
                  CallInst *,
                  ARGS(Value *Dst,
                       MaybeAlign DstAlign,
                       Value *Src,
                       MaybeAlign SrcAlign,
                       uint64_t Size,
                       bool isVolatile = false,
                       MDNode *TBAATag = nullptr,
                       MDNode *TBAAStructTag = nullptr,
                       MDNode *ScopeTag = nullptr,
                       MDNode *NoAliasTag = nullptr),
                  ARGS(Dst,
                       DstAlign,
                       Src,
                       SrcAlign,
                       Size,
                       isVolatile,
                       TBAATag,
                       TBAAStructTag,
                       ScopeTag,
                       NoAliasTag));
  EMBED_THE_CHECK(CreateMemTransferInst,
                  CallInst *,
                  ARGS(Intrinsic ::ID IntrID,
                       Value *Dst,
                       MaybeAlign DstAlign,
                       Value *Src,
                       MaybeAlign SrcAlign,
                       Value *Size,
                       bool isVolatile = false,
                       MDNode *TBAATag = nullptr,
                       MDNode *TBAAStructTag = nullptr,
                       MDNode *ScopeTag = nullptr,
                       MDNode *NoAliasTag = nullptr),
                  ARGS(IntrID,
                       Dst,
                       DstAlign,
                       Src,
                       SrcAlign,
                       Size,
                       isVolatile,
                       TBAATag,
                       TBAAStructTag,
                       ScopeTag,
                       NoAliasTag));
  EMBED_THE_CHECK(CreateMemCpy,
                  CallInst *,
                  ARGS(Value *Dst,
                       MaybeAlign DstAlign,
                       Value *Src,
                       MaybeAlign SrcAlign,
                       Value *Size,
                       bool isVolatile = false,
                       MDNode *TBAATag = nullptr,
                       MDNode *TBAAStructTag = nullptr,
                       MDNode *ScopeTag = nullptr,
                       MDNode *NoAliasTag = nullptr),
                  ARGS(Dst,
                       DstAlign,
                       Src,
                       SrcAlign,
                       Size,
                       isVolatile,
                       TBAATag,
                       TBAAStructTag,
                       ScopeTag,
                       NoAliasTag));
  EMBED_THE_CHECK(CreateMemCpyInline,
                  CallInst *,
                  ARGS(Value *Dst,
                       MaybeAlign DstAlign,
                       Value *Src,
                       MaybeAlign SrcAlign,
                       Value *Size,
                       bool IsVolatile = false,
                       MDNode *TBAATag = nullptr,
                       MDNode *TBAAStructTag = nullptr,
                       MDNode *ScopeTag = nullptr,
                       MDNode *NoAliasTag = nullptr),
                  ARGS(Dst,
                       DstAlign,
                       Src,
                       SrcAlign,
                       Size,
                       IsVolatile,
                       TBAATag,
                       TBAAStructTag,
                       ScopeTag,
                       NoAliasTag));
  EMBED_THE_CHECK(CreateElementUnorderedAtomicMemCpy,
                  CallInst *,
                  ARGS(Value *Dst,
                       Align DstAlign,
                       Value *Src,
                       Align SrcAlign,
                       Value *Size,
                       uint32_t ElementSize,
                       MDNode *TBAATag = nullptr,
                       MDNode *TBAAStructTag = nullptr,
                       MDNode *ScopeTag = nullptr,
                       MDNode *NoAliasTag = nullptr),
                  ARGS(Dst,
                       DstAlign,
                       Src,
                       SrcAlign,
                       Size,
                       ElementSize,
                       TBAATag,
                       TBAAStructTag,
                       ScopeTag,
                       NoAliasTag));
  EMBED_THE_CHECK(CreateMemMove,
                  CallInst *,
                  ARGS(Value *Dst,
                       MaybeAlign DstAlign,
                       Value *Src,
                       MaybeAlign SrcAlign,
                       uint64_t Size,
                       bool isVolatile = false,
                       MDNode *TBAATag = nullptr,
                       MDNode *ScopeTag = nullptr,
                       MDNode *NoAliasTag = nullptr),
                  ARGS(Dst,
                       DstAlign,
                       Src,
                       SrcAlign,
                       Size,
                       isVolatile,
                       TBAATag,
                       ScopeTag,
                       NoAliasTag));
  EMBED_THE_CHECK(CreateMemMove,
                  CallInst *,
                  ARGS(Value *Dst,
                       MaybeAlign DstAlign,
                       Value *Src,
                       MaybeAlign SrcAlign,
                       Value *Size,
                       bool isVolatile = false,
                       MDNode *TBAATag = nullptr,
                       MDNode *ScopeTag = nullptr,
                       MDNode *NoAliasTag = nullptr),
                  ARGS(Dst,
                       DstAlign,
                       Src,
                       SrcAlign,
                       Size,
                       isVolatile,
                       TBAATag,
                       ScopeTag,
                       NoAliasTag));
  EMBED_THE_CHECK(CreateElementUnorderedAtomicMemMove,
                  CallInst *,
                  ARGS(Value *Dst,
                       Align DstAlign,
                       Value *Src,
                       Align SrcAlign,
                       Value *Size,
                       uint32_t ElementSize,
                       MDNode *TBAATag = nullptr,
                       MDNode *TBAAStructTag = nullptr,
                       MDNode *ScopeTag = nullptr,
                       MDNode *NoAliasTag = nullptr),
                  ARGS(Dst,
                       DstAlign,
                       Src,
                       SrcAlign,
                       Size,
                       ElementSize,
                       TBAATag,
                       TBAAStructTag,
                       ScopeTag,
                       NoAliasTag));
  EMBED_THE_CHECK(CreateFAddReduce,
                  CallInst *,
                  ARGS(Value *Acc, Value *Src),
                  ARGS(Acc, Src));
  EMBED_THE_CHECK(CreateFMulReduce,
                  CallInst *,
                  ARGS(Value *Acc, Value *Src),
                  ARGS(Acc, Src));
  EMBED_THE_CHECK(CreateAddReduce, CallInst *, ARGS(Value *Src), ARGS(Src));
  EMBED_THE_CHECK(CreateMulReduce, CallInst *, ARGS(Value *Src), ARGS(Src));
  EMBED_THE_CHECK(CreateAndReduce, CallInst *, ARGS(Value *Src), ARGS(Src));
  EMBED_THE_CHECK(CreateOrReduce, CallInst *, ARGS(Value *Src), ARGS(Src));
  EMBED_THE_CHECK(CreateXorReduce, CallInst *, ARGS(Value *Src), ARGS(Src));
  EMBED_THE_CHECK(CreateIntMaxReduce,
                  CallInst *,
                  ARGS(Value *Src, bool IsSigned = false),
                  ARGS(Src, IsSigned));
  EMBED_THE_CHECK(CreateIntMinReduce,
                  CallInst *,
                  ARGS(Value *Src, bool IsSigned = false),
                  ARGS(Src, IsSigned));
  EMBED_THE_CHECK(CreateFPMaxReduce, CallInst *, ARGS(Value *Src), ARGS(Src));
  EMBED_THE_CHECK(CreateFPMinReduce, CallInst *, ARGS(Value *Src), ARGS(Src));
  EMBED_THE_CHECK(CreateLifetimeStart,
                  CallInst *,
                  ARGS(Value *Ptr, ConstantInt *Size = nullptr),
                  ARGS(Ptr, Size));
  EMBED_THE_CHECK(CreateLifetimeEnd,
                  CallInst *,
                  ARGS(Value *Ptr, ConstantInt *Size = nullptr),
                  ARGS(Ptr, Size));
  EMBED_THE_CHECK(CreateInvariantStart,
                  CallInst *,
                  ARGS(Value *Ptr, ConstantInt *Size = nullptr),
                  ARGS(Ptr, Size));
  EMBED_THE_CHECK(CreateThreadLocalAddress,
                  CallInst *,
                  ARGS(Value *Ptr),
                  ARGS(Ptr));
  EMBED_THE_CHECK(CreateMaskedLoad,
                  CallInst *,
                  ARGS(Type *Ty,
                       Value *Ptr,
                       Align Alignment,
                       Value *Mask,
                       Value *PassThru = nullptr,
                       const Twine &Name = ""),
                  ARGS(Ty, Ptr, Alignment, Mask, PassThru, Name));
  EMBED_THE_CHECK(CreateMaskedStore,
                  CallInst *,
                  ARGS(Value *Val, Value *Ptr, Align Alignment, Value *Mask),
                  ARGS(Val, Ptr, Alignment, Mask));
  EMBED_THE_CHECK(CreateMaskedGather,
                  CallInst *,
                  ARGS(Type *Ty,
                       Value *Ptrs,
                       Align Alignment,
                       Value *Mask = nullptr,
                       Value *PassThru = nullptr,
                       const Twine &Name = ""),
                  ARGS(Ty, Ptrs, Alignment, Mask, PassThru, Name));
  EMBED_THE_CHECK(CreateMaskedScatter,
                  CallInst *,
                  ARGS(Value *Val,
                       Value *Ptrs,
                       Align Alignment,
                       Value *Mask = nullptr),
                  ARGS(Val, Ptrs, Alignment, Mask));
  EMBED_THE_CHECK(CreateMaskedExpandLoad,
                  CallInst *,
                  ARGS(Type *Ty,
                       Value *Ptr,
                       Value *Mask = nullptr,
                       Value *PassThru = nullptr,
                       const Twine &Name = ""),
                  ARGS(Ty, Ptr, Mask, PassThru, Name));
  EMBED_THE_CHECK(CreateMaskedCompressStore,
                  CallInst *,
                  ARGS(Value *Val, Value *Ptr, Value *Mask = nullptr),
                  ARGS(Val, Ptr, Mask));
  EMBED_THE_CHECK(CreateAssumption,
                  CallInst *,
                  ARGS(Value *Cond,
                       ArrayRef<OperandBundleDef> OpBundles = std::nullopt),
                  ARGS(Cond, OpBundles));
  EMBED_THE_CHECK(CreateNoAliasScopeDeclaration,
                  Instruction *,
                  ARGS(Value *Scope),
                  ARGS(Scope));
  EMBED_THE_CHECK(CreateNoAliasScopeDeclaration,
                  Instruction *,
                  ARGS(MDNode *ScopeTag),
                  ARGS(ScopeTag));
  EMBED_THE_CHECK(CreateGCStatepointCall,
                  CallInst *,
                  ARGS(uint64_t ID,
                       uint32_t NumPatchBytes,
                       FunctionCallee ActualCallee,
                       ArrayRef<Value *> CallArgs,
                       std ::optional<ArrayRef<Value *>> DeoptArgs,
                       ArrayRef<Value *> GCArgs,
                       const Twine &Name = ""),
                  ARGS(ID,
                       NumPatchBytes,
                       ActualCallee,
                       CallArgs,
                       DeoptArgs,
                       GCArgs,
                       Name));
  EMBED_THE_CHECK(CreateGCStatepointCall,
                  CallInst *,
                  ARGS(uint64_t ID,
                       uint32_t NumPatchBytes,
                       FunctionCallee ActualCallee,
                       uint32_t Flags,
                       ArrayRef<Value *> CallArgs,
                       std ::optional<ArrayRef<Use>> TransitionArgs,
                       std ::optional<ArrayRef<Use>> DeoptArgs,
                       ArrayRef<Value *> GCArgs,
                       const Twine &Name = ""),
                  ARGS(ID,
                       NumPatchBytes,
                       ActualCallee,
                       Flags,
                       CallArgs,
                       TransitionArgs,
                       DeoptArgs,
                       GCArgs,
                       Name));
  EMBED_THE_CHECK(CreateGCStatepointCall,
                  CallInst *,
                  ARGS(uint64_t ID,
                       uint32_t NumPatchBytes,
                       FunctionCallee ActualCallee,
                       ArrayRef<Use> CallArgs,
                       std ::optional<ArrayRef<Value *>> DeoptArgs,
                       ArrayRef<Value *> GCArgs,
                       const Twine &Name = ""),
                  ARGS(ID,
                       NumPatchBytes,
                       ActualCallee,
                       CallArgs,
                       DeoptArgs,
                       GCArgs,
                       Name));
  EMBED_THE_CHECK(CreateGCStatepointInvoke,
                  InvokeInst *,
                  ARGS(uint64_t ID,
                       uint32_t NumPatchBytes,
                       FunctionCallee ActualInvokee,
                       BasicBlock *NormalDest,
                       BasicBlock *UnwindDest,
                       ArrayRef<Value *> InvokeArgs,
                       std ::optional<ArrayRef<Value *>> DeoptArgs,
                       ArrayRef<Value *> GCArgs,
                       const Twine &Name = ""),
                  ARGS(ID,
                       NumPatchBytes,
                       ActualInvokee,
                       NormalDest,
                       UnwindDest,
                       InvokeArgs,
                       DeoptArgs,
                       GCArgs,
                       Name));
  EMBED_THE_CHECK(CreateGCStatepointInvoke,
                  InvokeInst *,
                  ARGS(uint64_t ID,
                       uint32_t NumPatchBytes,
                       FunctionCallee ActualInvokee,
                       BasicBlock *NormalDest,
                       BasicBlock *UnwindDest,
                       uint32_t Flags,
                       ArrayRef<Value *> InvokeArgs,
                       std ::optional<ArrayRef<Use>> TransitionArgs,
                       std ::optional<ArrayRef<Use>> DeoptArgs,
                       ArrayRef<Value *> GCArgs,
                       const Twine &Name = ""),
                  ARGS(ID,
                       NumPatchBytes,
                       ActualInvokee,
                       NormalDest,
                       UnwindDest,
                       Flags,
                       InvokeArgs,
                       TransitionArgs,
                       DeoptArgs,
                       GCArgs,
                       Name));
  EMBED_THE_CHECK(CreateGCStatepointInvoke,
                  InvokeInst *,
                  ARGS(uint64_t ID,
                       uint32_t NumPatchBytes,
                       FunctionCallee ActualInvokee,
                       BasicBlock *NormalDest,
                       BasicBlock *UnwindDest,
                       ArrayRef<Use> InvokeArgs,
                       std ::optional<ArrayRef<Value *>> DeoptArgs,
                       ArrayRef<Value *> GCArgs,
                       const Twine &Name = ""),
                  ARGS(ID,
                       NumPatchBytes,
                       ActualInvokee,
                       NormalDest,
                       UnwindDest,
                       InvokeArgs,
                       DeoptArgs,
                       GCArgs,
                       Name));
  EMBED_THE_CHECK(CreateGCResult,
                  CallInst *,
                  ARGS(Instruction *Statepoint,
                       Type *ResultType,
                       const Twine &Name = ""),
                  ARGS(Statepoint, ResultType, Name));
  EMBED_THE_CHECK(CreateGCRelocate,
                  CallInst *,
                  ARGS(Instruction *Statepoint,
                       int BaseOffset,
                       int DerivedOffset,
                       Type *ResultType,
                       const Twine &Name = ""),
                  ARGS(Statepoint,
                       BaseOffset,
                       DerivedOffset,
                       ResultType,
                       Name));
  EMBED_THE_CHECK(CreateGCGetPointerBase,
                  CallInst *,
                  ARGS(Value *DerivedPtr, const Twine &Name = ""),
                  ARGS(DerivedPtr, Name));
  EMBED_THE_CHECK(CreateGCGetPointerOffset,
                  CallInst *,
                  ARGS(Value *DerivedPtr, const Twine &Name = ""),
                  ARGS(DerivedPtr, Name));
  EMBED_THE_CHECK(CreateVScale,
                  Value *,
                  ARGS(Constant *Scaling, const Twine &Name = ""),
                  ARGS(Scaling, Name));
  EMBED_THE_CHECK(CreateStepVector,
                  Value *,
                  ARGS(Type *DstType, const Twine &Name = ""),
                  ARGS(DstType, Name));
  EMBED_THE_CHECK(CreateUnaryIntrinsic,
                  CallInst *,
                  ARGS(Intrinsic ::ID ID,
                       Value *V,
                       Instruction *FMFSource = nullptr,
                       const Twine &Name = ""),
                  ARGS(ID, V, FMFSource, Name));
  EMBED_THE_CHECK(CreateBinaryIntrinsic,
                  CallInst *,
                  ARGS(Intrinsic ::ID ID,
                       Value *LHS,
                       Value *RHS,
                       Instruction *FMFSource = nullptr,
                       const Twine &Name = ""),
                  ARGS(ID, LHS, RHS, FMFSource, Name));
  EMBED_THE_CHECK(CreateIntrinsic,
                  CallInst *,
                  ARGS(Intrinsic ::ID ID,
                       ArrayRef<Type *> Types,
                       ArrayRef<Value *> Args,
                       Instruction *FMFSource = nullptr,
                       const Twine &Name = ""),
                  ARGS(ID, Types, Args, FMFSource, Name));
  EMBED_THE_CHECK(CreateIntrinsic,
                  CallInst *,
                  ARGS(Type *RetTy,
                       Intrinsic ::ID ID,
                       ArrayRef<Value *> Args,
                       Instruction *FMFSource = nullptr,
                       const Twine &Name = ""),
                  ARGS(RetTy, ID, Args, FMFSource, Name));
  EMBED_THE_CHECK(CreateMinNum,
                  CallInst *,
                  ARGS(Value *LHS, Value *RHS, const Twine &Name = ""),
                  ARGS(LHS, RHS, Name));
  EMBED_THE_CHECK(CreateMaxNum,
                  CallInst *,
                  ARGS(Value *LHS, Value *RHS, const Twine &Name = ""),
                  ARGS(LHS, RHS, Name));
  EMBED_THE_CHECK(CreateMinimum,
                  CallInst *,
                  ARGS(Value *LHS, Value *RHS, const Twine &Name = ""),
                  ARGS(LHS, RHS, Name));
  EMBED_THE_CHECK(CreateMaximum,
                  CallInst *,
                  ARGS(Value *LHS, Value *RHS, const Twine &Name = ""),
                  ARGS(LHS, RHS, Name));
  EMBED_THE_CHECK(CreateCopySign,
                  CallInst *,
                  ARGS(Value *LHS,
                       Value *RHS,
                       Instruction *FMFSource = nullptr,
                       const Twine &Name = ""),
                  ARGS(LHS, RHS, FMFSource, Name));
  EMBED_THE_CHECK(CreateArithmeticFence,
                  CallInst *,
                  ARGS(Value *Val, Type *DstType, const Twine &Name = ""),
                  ARGS(Val, DstType, Name));
  EMBED_THE_CHECK(CreateExtractVector,
                  CallInst *,
                  ARGS(Type *DstType,
                       Value *SrcVec,
                       Value *Idx,
                       const Twine &Name = ""),
                  ARGS(DstType, SrcVec, Idx, Name));
  EMBED_THE_CHECK(CreateInsertVector,
                  CallInst *,
                  ARGS(Type *DstType,
                       Value *SrcVec,
                       Value *SubVec,
                       Value *Idx,
                       const Twine &Name = ""),
                  ARGS(DstType, SrcVec, SubVec, Idx, Name));
  EMBED_THE_CHECK(CreateRetVoid, ReturnInst *, ARGS(), ARGS());
  EMBED_THE_CHECK(CreateRet, ReturnInst *, ARGS(Value *V), ARGS(V));
  EMBED_THE_CHECK(CreateAggregateRet,
                  ReturnInst *,
                  ARGS(Value *const *retVals, unsigned N),
                  ARGS(retVals, N));
  EMBED_THE_CHECK(CreateBr, BranchInst *, ARGS(BasicBlock *Dest), ARGS(Dest));
  EMBED_THE_CHECK(CreateCondBr,
                  BranchInst *,
                  ARGS(Value *Cond,
                       BasicBlock *True,
                       BasicBlock *False,
                       MDNode *BranchWeights = nullptr,
                       MDNode *Unpredictable = nullptr),
                  ARGS(Cond, True, False, BranchWeights, Unpredictable));
  EMBED_THE_CHECK(CreateCondBr,
                  BranchInst *,
                  ARGS(Value *Cond,
                       BasicBlock *True,
                       BasicBlock *False,
                       Instruction *MDSrc),
                  ARGS(Cond, True, False, MDSrc));
  EMBED_THE_CHECK(CreateSwitch,
                  SwitchInst *,
                  ARGS(Value *V,
                       BasicBlock *Dest,
                       unsigned NumCases = 10,
                       MDNode *BranchWeights = nullptr,
                       MDNode *Unpredictable = nullptr),
                  ARGS(V, Dest, NumCases, BranchWeights, Unpredictable));
  EMBED_THE_CHECK(CreateIndirectBr,
                  IndirectBrInst *,
                  ARGS(Value *Addr, unsigned NumDests = 10),
                  ARGS(Addr, NumDests));
  EMBED_THE_CHECK(CreateInvoke,
                  InvokeInst *,
                  ARGS(FunctionType *Ty,
                       Value *Callee,
                       BasicBlock *NormalDest,
                       BasicBlock *UnwindDest,
                       ArrayRef<Value *> Args,
                       ArrayRef<OperandBundleDef> OpBundles,
                       const Twine &Name = ""),
                  ARGS(Ty,
                       Callee,
                       NormalDest,
                       UnwindDest,
                       Args,
                       OpBundles,
                       Name));
  EMBED_THE_CHECK(CreateInvoke,
                  InvokeInst *,
                  ARGS(FunctionType *Ty,
                       Value *Callee,
                       BasicBlock *NormalDest,
                       BasicBlock *UnwindDest,
                       ArrayRef<Value *> Args = std::nullopt,
                       const Twine &Name = ""),
                  ARGS(Ty, Callee, NormalDest, UnwindDest, Args, Name));
  EMBED_THE_CHECK(CreateInvoke,
                  InvokeInst *,
                  ARGS(FunctionCallee Callee,
                       BasicBlock *NormalDest,
                       BasicBlock *UnwindDest,
                       ArrayRef<Value *> Args,
                       ArrayRef<OperandBundleDef> OpBundles,
                       const Twine &Name = ""),
                  ARGS(Callee, NormalDest, UnwindDest, Args, OpBundles, Name));
  EMBED_THE_CHECK(CreateInvoke,
                  InvokeInst *,
                  ARGS(FunctionCallee Callee,
                       BasicBlock *NormalDest,
                       BasicBlock *UnwindDest,
                       ArrayRef<Value *> Args = std::nullopt,
                       const Twine &Name = ""),
                  ARGS(Callee, NormalDest, UnwindDest, Args, Name));
  EMBED_THE_CHECK(CreateCallBr,
                  CallBrInst *,
                  ARGS(FunctionType *Ty,
                       Value *Callee,
                       BasicBlock *DefaultDest,
                       ArrayRef<BasicBlock *> IndirectDests,
                       ArrayRef<Value *> Args = std::nullopt,
                       const Twine &Name = ""),
                  ARGS(Ty, Callee, DefaultDest, IndirectDests, Args, Name));
  EMBED_THE_CHECK(CreateCallBr,
                  CallBrInst *,
                  ARGS(FunctionType *Ty,
                       Value *Callee,
                       BasicBlock *DefaultDest,
                       ArrayRef<BasicBlock *> IndirectDests,
                       ArrayRef<Value *> Args,
                       ArrayRef<OperandBundleDef> OpBundles,
                       const Twine &Name = ""),
                  ARGS(Ty,
                       Callee,
                       DefaultDest,
                       IndirectDests,
                       Args,
                       OpBundles,
                       Name));
  EMBED_THE_CHECK(CreateCallBr,
                  CallBrInst *,
                  ARGS(FunctionCallee Callee,
                       BasicBlock *DefaultDest,
                       ArrayRef<BasicBlock *> IndirectDests,
                       ArrayRef<Value *> Args = std::nullopt,
                       const Twine &Name = ""),
                  ARGS(Callee, DefaultDest, IndirectDests, Args, Name));
  EMBED_THE_CHECK(CreateCallBr,
                  CallBrInst *,
                  ARGS(FunctionCallee Callee,
                       BasicBlock *DefaultDest,
                       ArrayRef<BasicBlock *> IndirectDests,
                       ArrayRef<Value *> Args,
                       ArrayRef<OperandBundleDef> OpBundles,
                       const Twine &Name = ""),
                  ARGS(Callee,
                       DefaultDest,
                       IndirectDests,
                       Args,
                       OpBundles,
                       Name));
  EMBED_THE_CHECK(CreateResume, ResumeInst *, ARGS(Value *Exn), ARGS(Exn));
  EMBED_THE_CHECK(CreateCleanupRet,
                  CleanupReturnInst *,
                  ARGS(CleanupPadInst *CleanupPad,
                       BasicBlock *UnwindBB = nullptr),
                  ARGS(CleanupPad, UnwindBB));
  EMBED_THE_CHECK(CreateCatchSwitch,
                  CatchSwitchInst *,
                  ARGS(Value *ParentPad,
                       BasicBlock *UnwindBB,
                       unsigned NumHandlers,
                       const Twine &Name = ""),
                  ARGS(ParentPad, UnwindBB, NumHandlers, Name));
  EMBED_THE_CHECK(CreateCatchPad,
                  CatchPadInst *,
                  ARGS(Value *ParentPad,
                       ArrayRef<Value *> Args,
                       const Twine &Name = ""),
                  ARGS(ParentPad, Args, Name));
  EMBED_THE_CHECK(CreateCleanupPad,
                  CleanupPadInst *,
                  ARGS(Value *ParentPad,
                       ArrayRef<Value *> Args = std::nullopt,
                       const Twine &Name = ""),
                  ARGS(ParentPad, Args, Name));
  EMBED_THE_CHECK(CreateCatchRet,
                  CatchReturnInst *,
                  ARGS(CatchPadInst *CatchPad, BasicBlock *BB),
                  ARGS(CatchPad, BB));
  EMBED_THE_CHECK(CreateUnreachable, UnreachableInst *, ARGS(), ARGS());
  EMBED_THE_CHECK(CreateAdd,
                  Value *,
                  ARGS(Value *LHS,
                       Value *RHS,
                       const Twine &Name = "",
                       bool HasNUW = false,
                       bool HasNSW = false),
                  ARGS(LHS, RHS, Name, HasNUW, HasNSW));
  EMBED_THE_CHECK(CreateNSWAdd,
                  Value *,
                  ARGS(Value *LHS, Value *RHS, const Twine &Name = ""),
                  ARGS(LHS, RHS, Name));
  EMBED_THE_CHECK(CreateNUWAdd,
                  Value *,
                  ARGS(Value *LHS, Value *RHS, const Twine &Name = ""),
                  ARGS(LHS, RHS, Name));
  EMBED_THE_CHECK(CreateSub,
                  Value *,
                  ARGS(Value *LHS,
                       Value *RHS,
                       const Twine &Name = "",
                       bool HasNUW = false,
                       bool HasNSW = false),
                  ARGS(LHS, RHS, Name, HasNUW, HasNSW));
  EMBED_THE_CHECK(CreateNSWSub,
                  Value *,
                  ARGS(Value *LHS, Value *RHS, const Twine &Name = ""),
                  ARGS(LHS, RHS, Name));
  EMBED_THE_CHECK(CreateNUWSub,
                  Value *,
                  ARGS(Value *LHS, Value *RHS, const Twine &Name = ""),
                  ARGS(LHS, RHS, Name));
  EMBED_THE_CHECK(CreateMul,
                  Value *,
                  ARGS(Value *LHS,
                       Value *RHS,
                       const Twine &Name = "",
                       bool HasNUW = false,
                       bool HasNSW = false),
                  ARGS(LHS, RHS, Name, HasNUW, HasNSW));
  EMBED_THE_CHECK(CreateNSWMul,
                  Value *,
                  ARGS(Value *LHS, Value *RHS, const Twine &Name = ""),
                  ARGS(LHS, RHS, Name));
  EMBED_THE_CHECK(CreateNUWMul,
                  Value *,
                  ARGS(Value *LHS, Value *RHS, const Twine &Name = ""),
                  ARGS(LHS, RHS, Name));
  EMBED_THE_CHECK(CreateUDiv,
                  Value *,
                  ARGS(Value *LHS,
                       Value *RHS,
                       const Twine &Name = "",
                       bool isExact = false),
                  ARGS(LHS, RHS, Name, isExact));
  EMBED_THE_CHECK(CreateExactUDiv,
                  Value *,
                  ARGS(Value *LHS, Value *RHS, const Twine &Name = ""),
                  ARGS(LHS, RHS, Name));
  EMBED_THE_CHECK(CreateSDiv,
                  Value *,
                  ARGS(Value *LHS,
                       Value *RHS,
                       const Twine &Name = "",
                       bool isExact = false),
                  ARGS(LHS, RHS, Name, isExact));
  EMBED_THE_CHECK(CreateExactSDiv,
                  Value *,
                  ARGS(Value *LHS, Value *RHS, const Twine &Name = ""),
                  ARGS(LHS, RHS, Name));
  EMBED_THE_CHECK(CreateURem,
                  Value *,
                  ARGS(Value *LHS, Value *RHS, const Twine &Name = ""),
                  ARGS(LHS, RHS, Name));
  EMBED_THE_CHECK(CreateSRem,
                  Value *,
                  ARGS(Value *LHS, Value *RHS, const Twine &Name = ""),
                  ARGS(LHS, RHS, Name));
  EMBED_THE_CHECK(CreateShl,
                  Value *,
                  ARGS(Value *LHS,
                       Value *RHS,
                       const Twine &Name = "",
                       bool HasNUW = false,
                       bool HasNSW = false),
                  ARGS(LHS, RHS, Name, HasNUW, HasNSW));
  EMBED_THE_CHECK(CreateShl,
                  Value *,
                  ARGS(Value *LHS,
                       const APInt &RHS,
                       const Twine &Name = "",
                       bool HasNUW = false,
                       bool HasNSW = false),
                  ARGS(LHS, RHS, Name, HasNUW, HasNSW));
  EMBED_THE_CHECK(CreateShl,
                  Value *,
                  ARGS(Value *LHS,
                       uint64_t RHS,
                       const Twine &Name = "",
                       bool HasNUW = false,
                       bool HasNSW = false),
                  ARGS(LHS, RHS, Name, HasNUW, HasNSW));
  EMBED_THE_CHECK(CreateLShr,
                  Value *,
                  ARGS(Value *LHS,
                       Value *RHS,
                       const Twine &Name = "",
                       bool isExact = false),
                  ARGS(LHS, RHS, Name, isExact));
  EMBED_THE_CHECK(CreateLShr,
                  Value *,
                  ARGS(Value *LHS,
                       const APInt &RHS,
                       const Twine &Name = "",
                       bool isExact = false),
                  ARGS(LHS, RHS, Name, isExact));
  EMBED_THE_CHECK(CreateLShr,
                  Value *,
                  ARGS(Value *LHS,
                       uint64_t RHS,
                       const Twine &Name = "",
                       bool isExact = false),
                  ARGS(LHS, RHS, Name, isExact));
  EMBED_THE_CHECK(CreateAShr,
                  Value *,
                  ARGS(Value *LHS,
                       Value *RHS,
                       const Twine &Name = "",
                       bool isExact = false),
                  ARGS(LHS, RHS, Name, isExact));
  EMBED_THE_CHECK(CreateAShr,
                  Value *,
                  ARGS(Value *LHS,
                       const APInt &RHS,
                       const Twine &Name = "",
                       bool isExact = false),
                  ARGS(LHS, RHS, Name, isExact));
  EMBED_THE_CHECK(CreateAShr,
                  Value *,
                  ARGS(Value *LHS,
                       uint64_t RHS,
                       const Twine &Name = "",
                       bool isExact = false),
                  ARGS(LHS, RHS, Name, isExact));
  EMBED_THE_CHECK(CreateAnd,
                  Value *,
                  ARGS(Value *LHS, Value *RHS, const Twine &Name = ""),
                  ARGS(LHS, RHS, Name));
  EMBED_THE_CHECK(CreateAnd,
                  Value *,
                  ARGS(Value *LHS, const APInt &RHS, const Twine &Name = ""),
                  ARGS(LHS, RHS, Name));
  EMBED_THE_CHECK(CreateAnd,
                  Value *,
                  ARGS(Value *LHS, uint64_t RHS, const Twine &Name = ""),
                  ARGS(LHS, RHS, Name));
  EMBED_THE_CHECK(CreateAnd, Value *, ARGS(ArrayRef<Value *> Ops), ARGS(Ops));
  EMBED_THE_CHECK(CreateOr,
                  Value *,
                  ARGS(Value *LHS, Value *RHS, const Twine &Name = ""),
                  ARGS(LHS, RHS, Name));
  EMBED_THE_CHECK(CreateOr,
                  Value *,
                  ARGS(Value *LHS, const APInt &RHS, const Twine &Name = ""),
                  ARGS(LHS, RHS, Name));
  EMBED_THE_CHECK(CreateOr,
                  Value *,
                  ARGS(Value *LHS, uint64_t RHS, const Twine &Name = ""),
                  ARGS(LHS, RHS, Name));
  EMBED_THE_CHECK(CreateOr, Value *, ARGS(ArrayRef<Value *> Ops), ARGS(Ops));
  EMBED_THE_CHECK(CreateXor,
                  Value *,
                  ARGS(Value *LHS, Value *RHS, const Twine &Name = ""),
                  ARGS(LHS, RHS, Name));
  EMBED_THE_CHECK(CreateXor,
                  Value *,
                  ARGS(Value *LHS, const APInt &RHS, const Twine &Name = ""),
                  ARGS(LHS, RHS, Name));
  EMBED_THE_CHECK(CreateXor,
                  Value *,
                  ARGS(Value *LHS, uint64_t RHS, const Twine &Name = ""),
                  ARGS(LHS, RHS, Name));
  EMBED_THE_CHECK(CreateFAdd,
                  Value *,
                  ARGS(Value *L,
                       Value *R,
                       const Twine &Name = "",
                       MDNode *FPMD = nullptr),
                  ARGS(L, R, Name, FPMD));
  EMBED_THE_CHECK(CreateFAddFMF,
                  Value *,
                  ARGS(Value *L,
                       Value *R,
                       Instruction *FMFSource,
                       const Twine &Name = ""),
                  ARGS(L, R, FMFSource, Name));
  EMBED_THE_CHECK(CreateFSub,
                  Value *,
                  ARGS(Value *L,
                       Value *R,
                       const Twine &Name = "",
                       MDNode *FPMD = nullptr),
                  ARGS(L, R, Name, FPMD));
  EMBED_THE_CHECK(CreateFSubFMF,
                  Value *,
                  ARGS(Value *L,
                       Value *R,
                       Instruction *FMFSource,
                       const Twine &Name = ""),
                  ARGS(L, R, FMFSource, Name));
  EMBED_THE_CHECK(CreateFMul,
                  Value *,
                  ARGS(Value *L,
                       Value *R,
                       const Twine &Name = "",
                       MDNode *FPMD = nullptr),
                  ARGS(L, R, Name, FPMD));
  EMBED_THE_CHECK(CreateFMulFMF,
                  Value *,
                  ARGS(Value *L,
                       Value *R,
                       Instruction *FMFSource,
                       const Twine &Name = ""),
                  ARGS(L, R, FMFSource, Name));
  EMBED_THE_CHECK(CreateFDiv,
                  Value *,
                  ARGS(Value *L,
                       Value *R,
                       const Twine &Name = "",
                       MDNode *FPMD = nullptr),
                  ARGS(L, R, Name, FPMD));
  EMBED_THE_CHECK(CreateFDivFMF,
                  Value *,
                  ARGS(Value *L,
                       Value *R,
                       Instruction *FMFSource,
                       const Twine &Name = ""),
                  ARGS(L, R, FMFSource, Name));
  EMBED_THE_CHECK(CreateFRem,
                  Value *,
                  ARGS(Value *L,
                       Value *R,
                       const Twine &Name = "",
                       MDNode *FPMD = nullptr),
                  ARGS(L, R, Name, FPMD));
  EMBED_THE_CHECK(CreateFRemFMF,
                  Value *,
                  ARGS(Value *L,
                       Value *R,
                       Instruction *FMFSource,
                       const Twine &Name = ""),
                  ARGS(L, R, FMFSource, Name));
  EMBED_THE_CHECK(CreateBinOp,
                  Value *,
                  ARGS(Instruction ::BinaryOps Opc,
                       Value *LHS,
                       Value *RHS,
                       const Twine &Name = "",
                       MDNode *FPMathTag = nullptr),
                  ARGS(Opc, LHS, RHS, Name, FPMathTag));
  EMBED_THE_CHECK(CreateLogicalAnd,
                  Value *,
                  ARGS(Value *Cond1, Value *Cond2, const Twine &Name = ""),
                  ARGS(Cond1, Cond2, Name));
  EMBED_THE_CHECK(CreateLogicalOr,
                  Value *,
                  ARGS(Value *Cond1, Value *Cond2, const Twine &Name = ""),
                  ARGS(Cond1, Cond2, Name));
  EMBED_THE_CHECK(CreateLogicalOp,
                  Value *,
                  ARGS(Instruction ::BinaryOps Opc,
                       Value *Cond1,
                       Value *Cond2,
                       const Twine &Name = ""),
                  ARGS(Opc, Cond1, Cond2, Name));
  EMBED_THE_CHECK(CreateLogicalOr,
                  Value *,
                  ARGS(ArrayRef<Value *> Ops),
                  ARGS(Ops));
  EMBED_THE_CHECK(CreateConstrainedFPBinOp,
                  CallInst *,
                  ARGS(Intrinsic ::ID ID,
                       Value *L,
                       Value *R,
                       Instruction *FMFSource = nullptr,
                       const Twine &Name = "",
                       MDNode *FPMathTag = nullptr,
                       std ::optional<RoundingMode> Rounding = std::nullopt,
                       std ::optional<fp ::ExceptionBehavior> Except =
                         std::nullopt),
                  ARGS(ID, L, R, FMFSource, Name, FPMathTag, Rounding, Except));
  EMBED_THE_CHECK(CreateNeg,
                  Value *,
                  ARGS(Value *V,
                       const Twine &Name = "",
                       bool HasNUW = false,
                       bool HasNSW = false),
                  ARGS(V, Name, HasNUW, HasNSW));
  EMBED_THE_CHECK(CreateNSWNeg,
                  Value *,
                  ARGS(Value *V, const Twine &Name = ""),
                  ARGS(V, Name));
  EMBED_THE_CHECK(CreateNUWNeg,
                  Value *,
                  ARGS(Value *V, const Twine &Name = ""),
                  ARGS(V, Name));
  EMBED_THE_CHECK(CreateFNeg,
                  Value *,
                  ARGS(Value *V,
                       const Twine &Name = "",
                       MDNode *FPMathTag = nullptr),
                  ARGS(V, Name, FPMathTag));
  EMBED_THE_CHECK(CreateFNegFMF,
                  Value *,
                  ARGS(Value *V,
                       Instruction *FMFSource,
                       const Twine &Name = ""),
                  ARGS(V, FMFSource, Name));
  EMBED_THE_CHECK(CreateNot,
                  Value *,
                  ARGS(Value *V, const Twine &Name = ""),
                  ARGS(V, Name));
  EMBED_THE_CHECK(CreateUnOp,
                  Value *,
                  ARGS(Instruction ::UnaryOps Opc,
                       Value *V,
                       const Twine &Name = "",
                       MDNode *FPMathTag = nullptr),
                  ARGS(Opc, V, Name, FPMathTag));
  EMBED_THE_CHECK(CreateNAryOp,
                  Value *,
                  ARGS(unsigned Opc,
                       ArrayRef<Value *> Ops,
                       const Twine &Name = "",
                       MDNode *FPMathTag = nullptr),
                  ARGS(Opc, Ops, Name, FPMathTag));
  EMBED_THE_CHECK(CreateAlloca,
                  AllocaInst *,
                  ARGS(Type *Ty,
                       unsigned AddrSpace,
                       Value *ArraySize = nullptr,
                       const Twine &Name = ""),
                  ARGS(Ty, AddrSpace, ArraySize, Name));
  EMBED_THE_CHECK(CreateAlloca,
                  AllocaInst *,
                  ARGS(Type *Ty,
                       Value *ArraySize = nullptr,
                       const Twine &Name = ""),
                  ARGS(Ty, ArraySize, Name));
  EMBED_THE_CHECK(CreateLoad,
                  LoadInst *,
                  ARGS(Type *Ty, Value *Ptr, const char *Name),
                  ARGS(Ty, Ptr, Name));
  EMBED_THE_CHECK(CreateLoad,
                  LoadInst *,
                  ARGS(Type *Ty, Value *Ptr, const Twine &Name = ""),
                  ARGS(Ty, Ptr, Name));
  EMBED_THE_CHECK(CreateLoad,
                  LoadInst *,
                  ARGS(Type *Ty,
                       Value *Ptr,
                       bool isVolatile,
                       const Twine &Name = ""),
                  ARGS(Ty, Ptr, isVolatile, Name));
  EMBED_THE_CHECK(CreateStore,
                  StoreInst *,
                  ARGS(Value *Val, Value *Ptr, bool isVolatile = false),
                  ARGS(Val, Ptr, isVolatile));
  EMBED_THE_CHECK(CreateAlignedLoad,
                  LoadInst *,
                  ARGS(Type *Ty,
                       Value *Ptr,
                       MaybeAlign Align,
                       const char *Name),
                  ARGS(Ty, Ptr, Align, Name));
  EMBED_THE_CHECK(CreateAlignedLoad,
                  LoadInst *,
                  ARGS(Type *Ty,
                       Value *Ptr,
                       MaybeAlign Align,
                       const Twine &Name = ""),
                  ARGS(Ty, Ptr, Align, Name));
  EMBED_THE_CHECK(CreateAlignedLoad,
                  LoadInst *,
                  ARGS(Type *Ty,
                       Value *Ptr,
                       MaybeAlign Align,
                       bool isVolatile,
                       const Twine &Name = ""),
                  ARGS(Ty, Ptr, Align, isVolatile, Name));
  EMBED_THE_CHECK(CreateAlignedStore,
                  StoreInst *,
                  ARGS(Value *Val,
                       Value *Ptr,
                       MaybeAlign Align,
                       bool isVolatile = false),
                  ARGS(Val, Ptr, Align, isVolatile));
  EMBED_THE_CHECK(CreateFence,
                  FenceInst *,
                  ARGS(AtomicOrdering Ordering,
                       SyncScope ::ID SSID = SyncScope::System,
                       const Twine &Name = ""),
                  ARGS(Ordering, SSID, Name));
  EMBED_THE_CHECK(CreateAtomicCmpXchg,
                  AtomicCmpXchgInst *,
                  ARGS(Value *Ptr,
                       Value *Cmp,
                       Value *New,
                       MaybeAlign Align,
                       AtomicOrdering SuccessOrdering,
                       AtomicOrdering FailureOrdering,
                       SyncScope ::ID SSID = SyncScope::System),
                  ARGS(Ptr,
                       Cmp,
                       New,
                       Align,
                       SuccessOrdering,
                       FailureOrdering,
                       SSID));
  EMBED_THE_CHECK(CreateAtomicRMW,
                  AtomicRMWInst *,
                  ARGS(AtomicRMWInst ::BinOp Op,
                       Value *Ptr,
                       Value *Val,
                       MaybeAlign Align,
                       AtomicOrdering Ordering,
                       SyncScope ::ID SSID = SyncScope::System),
                  ARGS(Op, Ptr, Val, Align, Ordering, SSID));
  EMBED_THE_CHECK(CreateGEP,
                  Value *,
                  ARGS(Type *Ty,
                       Value *Ptr,
                       ArrayRef<Value *> IdxList,
                       const Twine &Name = "",
                       bool IsInBounds = false),
                  ARGS(Ty, Ptr, IdxList, Name, IsInBounds));
  EMBED_THE_CHECK(CreateInBoundsGEP,
                  Value *,
                  ARGS(Type *Ty,
                       Value *Ptr,
                       ArrayRef<Value *> IdxList,
                       const Twine &Name = ""),
                  ARGS(Ty, Ptr, IdxList, Name));
  EMBED_THE_CHECK(CreateConstGEP1_32,
                  Value *,
                  ARGS(Type *Ty,
                       Value *Ptr,
                       unsigned Idx0,
                       const Twine &Name = ""),
                  ARGS(Ty, Ptr, Idx0, Name));
  EMBED_THE_CHECK(CreateConstInBoundsGEP1_32,
                  Value *,
                  ARGS(Type *Ty,
                       Value *Ptr,
                       unsigned Idx0,
                       const Twine &Name = ""),
                  ARGS(Ty, Ptr, Idx0, Name));
  EMBED_THE_CHECK(CreateConstGEP2_32,
                  Value *,
                  ARGS(Type *Ty,
                       Value *Ptr,
                       unsigned Idx0,
                       unsigned Idx1,
                       const Twine &Name = ""),
                  ARGS(Ty, Ptr, Idx0, Idx1, Name));
  EMBED_THE_CHECK(CreateConstInBoundsGEP2_32,
                  Value *,
                  ARGS(Type *Ty,
                       Value *Ptr,
                       unsigned Idx0,
                       unsigned Idx1,
                       const Twine &Name = ""),
                  ARGS(Ty, Ptr, Idx0, Idx1, Name));
  EMBED_THE_CHECK(CreateConstGEP1_64,
                  Value *,
                  ARGS(Type *Ty,
                       Value *Ptr,
                       uint64_t Idx0,
                       const Twine &Name = ""),
                  ARGS(Ty, Ptr, Idx0, Name));
  EMBED_THE_CHECK(CreateConstInBoundsGEP1_64,
                  Value *,
                  ARGS(Type *Ty,
                       Value *Ptr,
                       uint64_t Idx0,
                       const Twine &Name = ""),
                  ARGS(Ty, Ptr, Idx0, Name));
  EMBED_THE_CHECK(CreateConstGEP2_64,
                  Value *,
                  ARGS(Type *Ty,
                       Value *Ptr,
                       uint64_t Idx0,
                       uint64_t Idx1,
                       const Twine &Name = ""),
                  ARGS(Ty, Ptr, Idx0, Idx1, Name));
  EMBED_THE_CHECK(CreateConstInBoundsGEP2_64,
                  Value *,
                  ARGS(Type *Ty,
                       Value *Ptr,
                       uint64_t Idx0,
                       uint64_t Idx1,
                       const Twine &Name = ""),
                  ARGS(Ty, Ptr, Idx0, Idx1, Name));
  EMBED_THE_CHECK(CreateStructGEP,
                  Value *,
                  ARGS(Type *Ty,
                       Value *Ptr,
                       unsigned Idx,
                       const Twine &Name = ""),
                  ARGS(Ty, Ptr, Idx, Name));
  EMBED_THE_CHECK(CreateGlobalStringPtr,
                  Constant *,
                  ARGS(StringRef Str,
                       const Twine &Name = "",
                       unsigned AddressSpace = 0,
                       Module *M = nullptr),
                  ARGS(Str, Name, AddressSpace, M));
  EMBED_THE_CHECK(CreateTrunc,
                  Value *,
                  ARGS(Value *V, Type *DestTy, const Twine &Name = ""),
                  ARGS(V, DestTy, Name));
  EMBED_THE_CHECK(CreateZExt,
                  Value *,
                  ARGS(Value *V, Type *DestTy, const Twine &Name = ""),
                  ARGS(V, DestTy, Name));
  EMBED_THE_CHECK(CreateSExt,
                  Value *,
                  ARGS(Value *V, Type *DestTy, const Twine &Name = ""),
                  ARGS(V, DestTy, Name));
  EMBED_THE_CHECK(CreateZExtOrTrunc,
                  Value *,
                  ARGS(Value *V, Type *DestTy, const Twine &Name = ""),
                  ARGS(V, DestTy, Name));
  EMBED_THE_CHECK(CreateSExtOrTrunc,
                  Value *,
                  ARGS(Value *V, Type *DestTy, const Twine &Name = ""),
                  ARGS(V, DestTy, Name));
  EMBED_THE_CHECK(CreateFPToUI,
                  Value *,
                  ARGS(Value *V, Type *DestTy, const Twine &Name = ""),
                  ARGS(V, DestTy, Name));
  EMBED_THE_CHECK(CreateFPToSI,
                  Value *,
                  ARGS(Value *V, Type *DestTy, const Twine &Name = ""),
                  ARGS(V, DestTy, Name));
  EMBED_THE_CHECK(CreateUIToFP,
                  Value *,
                  ARGS(Value *V, Type *DestTy, const Twine &Name = ""),
                  ARGS(V, DestTy, Name));
  EMBED_THE_CHECK(CreateSIToFP,
                  Value *,
                  ARGS(Value *V, Type *DestTy, const Twine &Name = ""),
                  ARGS(V, DestTy, Name));
  EMBED_THE_CHECK(CreateFPTrunc,
                  Value *,
                  ARGS(Value *V, Type *DestTy, const Twine &Name = ""),
                  ARGS(V, DestTy, Name));
  EMBED_THE_CHECK(CreateFPExt,
                  Value *,
                  ARGS(Value *V, Type *DestTy, const Twine &Name = ""),
                  ARGS(V, DestTy, Name));
  EMBED_THE_CHECK(CreatePtrToInt,
                  Value *,
                  ARGS(Value *V, Type *DestTy, const Twine &Name = ""),
                  ARGS(V, DestTy, Name));
  EMBED_THE_CHECK(CreateIntToPtr,
                  Value *,
                  ARGS(Value *V, Type *DestTy, const Twine &Name = ""),
                  ARGS(V, DestTy, Name));
  EMBED_THE_CHECK(CreateBitCast,
                  Value *,
                  ARGS(Value *V, Type *DestTy, const Twine &Name = ""),
                  ARGS(V, DestTy, Name));
  EMBED_THE_CHECK(CreateAddrSpaceCast,
                  Value *,
                  ARGS(Value *V, Type *DestTy, const Twine &Name = ""),
                  ARGS(V, DestTy, Name));
  EMBED_THE_CHECK(CreateZExtOrBitCast,
                  Value *,
                  ARGS(Value *V, Type *DestTy, const Twine &Name = ""),
                  ARGS(V, DestTy, Name));
  EMBED_THE_CHECK(CreateSExtOrBitCast,
                  Value *,
                  ARGS(Value *V, Type *DestTy, const Twine &Name = ""),
                  ARGS(V, DestTy, Name));
  EMBED_THE_CHECK(CreateTruncOrBitCast,
                  Value *,
                  ARGS(Value *V, Type *DestTy, const Twine &Name = ""),
                  ARGS(V, DestTy, Name));
  EMBED_THE_CHECK(CreateCast,
                  Value *,
                  ARGS(Instruction ::CastOps Op,
                       Value *V,
                       Type *DestTy,
                       const Twine &Name = ""),
                  ARGS(Op, V, DestTy, Name));
  EMBED_THE_CHECK(CreatePointerCast,
                  Value *,
                  ARGS(Value *V, Type *DestTy, const Twine &Name = ""),
                  ARGS(V, DestTy, Name));
  EMBED_THE_CHECK(CreatePointerBitCastOrAddrSpaceCast,
                  Value *,
                  ARGS(Value *V, Type *DestTy, const Twine &Name = ""),
                  ARGS(V, DestTy, Name));
  EMBED_THE_CHECK(CreateIntCast,
                  Value *,
                  ARGS(Value *V,
                       Type *DestTy,
                       bool isSigned,
                       const Twine &Name = ""),
                  ARGS(V, DestTy, isSigned, Name));
  EMBED_THE_CHECK(CreateBitOrPointerCast,
                  Value *,
                  ARGS(Value *V, Type *DestTy, const Twine &Name = ""),
                  ARGS(V, DestTy, Name));
  EMBED_THE_CHECK(CreateFPCast,
                  Value *,
                  ARGS(Value *V, Type *DestTy, const Twine &Name = ""),
                  ARGS(V, DestTy, Name));
  EMBED_THE_CHECK(CreateConstrainedFPCast,
                  CallInst *,
                  ARGS(Intrinsic ::ID ID,
                       Value *V,
                       Type *DestTy,
                       Instruction *FMFSource = nullptr,
                       const Twine &Name = "",
                       MDNode *FPMathTag = nullptr,
                       std ::optional<RoundingMode> Rounding = std::nullopt,
                       std ::optional<fp ::ExceptionBehavior> Except =
                         std::nullopt),
                  ARGS(ID,
                       V,
                       DestTy,
                       FMFSource,
                       Name,
                       FPMathTag,
                       Rounding,
                       Except));
  EMBED_THE_CHECK(CreateICmpEQ,
                  Value *,
                  ARGS(Value *LHS, Value *RHS, const Twine &Name = ""),
                  ARGS(LHS, RHS, Name));
  EMBED_THE_CHECK(CreateICmpNE,
                  Value *,
                  ARGS(Value *LHS, Value *RHS, const Twine &Name = ""),
                  ARGS(LHS, RHS, Name));
  EMBED_THE_CHECK(CreateICmpUGT,
                  Value *,
                  ARGS(Value *LHS, Value *RHS, const Twine &Name = ""),
                  ARGS(LHS, RHS, Name));
  EMBED_THE_CHECK(CreateICmpUGE,
                  Value *,
                  ARGS(Value *LHS, Value *RHS, const Twine &Name = ""),
                  ARGS(LHS, RHS, Name));
  EMBED_THE_CHECK(CreateICmpULT,
                  Value *,
                  ARGS(Value *LHS, Value *RHS, const Twine &Name = ""),
                  ARGS(LHS, RHS, Name));
  EMBED_THE_CHECK(CreateICmpULE,
                  Value *,
                  ARGS(Value *LHS, Value *RHS, const Twine &Name = ""),
                  ARGS(LHS, RHS, Name));
  EMBED_THE_CHECK(CreateICmpSGT,
                  Value *,
                  ARGS(Value *LHS, Value *RHS, const Twine &Name = ""),
                  ARGS(LHS, RHS, Name));
  EMBED_THE_CHECK(CreateICmpSGE,
                  Value *,
                  ARGS(Value *LHS, Value *RHS, const Twine &Name = ""),
                  ARGS(LHS, RHS, Name));
  EMBED_THE_CHECK(CreateICmpSLT,
                  Value *,
                  ARGS(Value *LHS, Value *RHS, const Twine &Name = ""),
                  ARGS(LHS, RHS, Name));
  EMBED_THE_CHECK(CreateICmpSLE,
                  Value *,
                  ARGS(Value *LHS, Value *RHS, const Twine &Name = ""),
                  ARGS(LHS, RHS, Name));
  EMBED_THE_CHECK(CreateFCmpOEQ,
                  Value *,
                  ARGS(Value *LHS,
                       Value *RHS,
                       const Twine &Name = "",
                       MDNode *FPMathTag = nullptr),
                  ARGS(LHS, RHS, Name, FPMathTag));
  EMBED_THE_CHECK(CreateFCmpOGT,
                  Value *,
                  ARGS(Value *LHS,
                       Value *RHS,
                       const Twine &Name = "",
                       MDNode *FPMathTag = nullptr),
                  ARGS(LHS, RHS, Name, FPMathTag));
  EMBED_THE_CHECK(CreateFCmpOGE,
                  Value *,
                  ARGS(Value *LHS,
                       Value *RHS,
                       const Twine &Name = "",
                       MDNode *FPMathTag = nullptr),
                  ARGS(LHS, RHS, Name, FPMathTag));
  EMBED_THE_CHECK(CreateFCmpOLT,
                  Value *,
                  ARGS(Value *LHS,
                       Value *RHS,
                       const Twine &Name = "",
                       MDNode *FPMathTag = nullptr),
                  ARGS(LHS, RHS, Name, FPMathTag));
  EMBED_THE_CHECK(CreateFCmpOLE,
                  Value *,
                  ARGS(Value *LHS,
                       Value *RHS,
                       const Twine &Name = "",
                       MDNode *FPMathTag = nullptr),
                  ARGS(LHS, RHS, Name, FPMathTag));
  EMBED_THE_CHECK(CreateFCmpONE,
                  Value *,
                  ARGS(Value *LHS,
                       Value *RHS,
                       const Twine &Name = "",
                       MDNode *FPMathTag = nullptr),
                  ARGS(LHS, RHS, Name, FPMathTag));
  EMBED_THE_CHECK(CreateFCmpORD,
                  Value *,
                  ARGS(Value *LHS,
                       Value *RHS,
                       const Twine &Name = "",
                       MDNode *FPMathTag = nullptr),
                  ARGS(LHS, RHS, Name, FPMathTag));
  EMBED_THE_CHECK(CreateFCmpUNO,
                  Value *,
                  ARGS(Value *LHS,
                       Value *RHS,
                       const Twine &Name = "",
                       MDNode *FPMathTag = nullptr),
                  ARGS(LHS, RHS, Name, FPMathTag));
  EMBED_THE_CHECK(CreateFCmpUEQ,
                  Value *,
                  ARGS(Value *LHS,
                       Value *RHS,
                       const Twine &Name = "",
                       MDNode *FPMathTag = nullptr),
                  ARGS(LHS, RHS, Name, FPMathTag));
  EMBED_THE_CHECK(CreateFCmpUGT,
                  Value *,
                  ARGS(Value *LHS,
                       Value *RHS,
                       const Twine &Name = "",
                       MDNode *FPMathTag = nullptr),
                  ARGS(LHS, RHS, Name, FPMathTag));
  EMBED_THE_CHECK(CreateFCmpUGE,
                  Value *,
                  ARGS(Value *LHS,
                       Value *RHS,
                       const Twine &Name = "",
                       MDNode *FPMathTag = nullptr),
                  ARGS(LHS, RHS, Name, FPMathTag));
  EMBED_THE_CHECK(CreateFCmpULT,
                  Value *,
                  ARGS(Value *LHS,
                       Value *RHS,
                       const Twine &Name = "",
                       MDNode *FPMathTag = nullptr),
                  ARGS(LHS, RHS, Name, FPMathTag));
  EMBED_THE_CHECK(CreateFCmpULE,
                  Value *,
                  ARGS(Value *LHS,
                       Value *RHS,
                       const Twine &Name = "",
                       MDNode *FPMathTag = nullptr),
                  ARGS(LHS, RHS, Name, FPMathTag));
  EMBED_THE_CHECK(CreateFCmpUNE,
                  Value *,
                  ARGS(Value *LHS,
                       Value *RHS,
                       const Twine &Name = "",
                       MDNode *FPMathTag = nullptr),
                  ARGS(LHS, RHS, Name, FPMathTag));
  EMBED_THE_CHECK(CreateICmp,
                  Value *,
                  ARGS(CmpInst ::Predicate P,
                       Value *LHS,
                       Value *RHS,
                       const Twine &Name = ""),
                  ARGS(P, LHS, RHS, Name));
  EMBED_THE_CHECK(CreateFCmp,
                  Value *,
                  ARGS(CmpInst ::Predicate P,
                       Value *LHS,
                       Value *RHS,
                       const Twine &Name = "",
                       MDNode *FPMathTag = nullptr),
                  ARGS(P, LHS, RHS, Name, FPMathTag));
  EMBED_THE_CHECK(CreateCmp,
                  Value *,
                  ARGS(CmpInst ::Predicate Pred,
                       Value *LHS,
                       Value *RHS,
                       const Twine &Name = "",
                       MDNode *FPMathTag = nullptr),
                  ARGS(Pred, LHS, RHS, Name, FPMathTag));
  EMBED_THE_CHECK(CreateFCmpS,
                  Value *,
                  ARGS(CmpInst ::Predicate P,
                       Value *LHS,
                       Value *RHS,
                       const Twine &Name = "",
                       MDNode *FPMathTag = nullptr),
                  ARGS(P, LHS, RHS, Name, FPMathTag));
  EMBED_THE_CHECK(CreateConstrainedFPCmp,
                  CallInst *,
                  ARGS(Intrinsic ::ID ID,
                       CmpInst ::Predicate P,
                       Value *L,
                       Value *R,
                       const Twine &Name = "",
                       std ::optional<fp ::ExceptionBehavior> Except =
                         std::nullopt),
                  ARGS(ID, P, L, R, Name, Except));
  EMBED_THE_CHECK(CreatePHI,
                  PHINode *,
                  ARGS(Type *Ty,
                       unsigned NumReservedValues,
                       const Twine &Name = ""),
                  ARGS(Ty, NumReservedValues, Name));
  EMBED_THE_CHECK(CreateCall,
                  CallInst *,
                  ARGS(FunctionType *FTy,
                       Value *Callee,
                       ArrayRef<Value *> Args = std::nullopt,
                       const Twine &Name = "",
                       MDNode *FPMathTag = nullptr),
                  ARGS(FTy, Callee, Args, Name, FPMathTag));
  EMBED_THE_CHECK(CreateCall,
                  CallInst *,
                  ARGS(FunctionType *FTy,
                       Value *Callee,
                       ArrayRef<Value *> Args,
                       ArrayRef<OperandBundleDef> OpBundles,
                       const Twine &Name = "",
                       MDNode *FPMathTag = nullptr),
                  ARGS(FTy, Callee, Args, OpBundles, Name, FPMathTag));
  EMBED_THE_CHECK(CreateCall,
                  CallInst *,
                  ARGS(FunctionCallee Callee,
                       ArrayRef<Value *> Args = std::nullopt,
                       const Twine &Name = "",
                       MDNode *FPMathTag = nullptr),
                  ARGS(Callee, Args, Name, FPMathTag));
  EMBED_THE_CHECK(CreateCall,
                  CallInst *,
                  ARGS(FunctionCallee Callee,
                       ArrayRef<Value *> Args,
                       ArrayRef<OperandBundleDef> OpBundles,
                       const Twine &Name = "",
                       MDNode *FPMathTag = nullptr),
                  ARGS(Callee, Args, OpBundles, Name, FPMathTag));
  EMBED_THE_CHECK(CreateConstrainedFPCall,
                  CallInst *,
                  ARGS(Function *Callee,
                       ArrayRef<Value *> Args,
                       const Twine &Name = "",
                       std ::optional<RoundingMode> Rounding = std::nullopt,
                       std ::optional<fp ::ExceptionBehavior> Except =
                         std::nullopt),
                  ARGS(Callee, Args, Name, Rounding, Except));
  EMBED_THE_CHECK(CreateSelect,
                  Value *,
                  ARGS(Value *C,
                       Value *True,
                       Value *False,
                       const Twine &Name = "",
                       Instruction *MDFrom = nullptr),
                  ARGS(C, True, False, Name, MDFrom));
  EMBED_THE_CHECK(CreateVAArg,
                  VAArgInst *,
                  ARGS(Value *List, Type *Ty, const Twine &Name = ""),
                  ARGS(List, Ty, Name));
  EMBED_THE_CHECK(CreateExtractElement,
                  Value *,
                  ARGS(Value *Vec, Value *Idx, const Twine &Name = ""),
                  ARGS(Vec, Idx, Name));
  EMBED_THE_CHECK(CreateExtractElement,
                  Value *,
                  ARGS(Value *Vec, uint64_t Idx, const Twine &Name = ""),
                  ARGS(Vec, Idx, Name));
  EMBED_THE_CHECK(CreateInsertElement,
                  Value *,
                  ARGS(Type *VecTy,
                       Value *NewElt,
                       Value *Idx,
                       const Twine &Name = ""),
                  ARGS(VecTy, NewElt, Idx, Name));
  EMBED_THE_CHECK(CreateInsertElement,
                  Value *,
                  ARGS(Type *VecTy,
                       Value *NewElt,
                       uint64_t Idx,
                       const Twine &Name = ""),
                  ARGS(VecTy, NewElt, Idx, Name));
  EMBED_THE_CHECK(CreateInsertElement,
                  Value *,
                  ARGS(Value *Vec,
                       Value *NewElt,
                       Value *Idx,
                       const Twine &Name = ""),
                  ARGS(Vec, NewElt, Idx, Name));
  EMBED_THE_CHECK(CreateInsertElement,
                  Value *,
                  ARGS(Value *Vec,
                       Value *NewElt,
                       uint64_t Idx,
                       const Twine &Name = ""),
                  ARGS(Vec, NewElt, Idx, Name));
  EMBED_THE_CHECK(CreateShuffleVector,
                  Value *,
                  ARGS(Value *V1,
                       Value *V2,
                       Value *Mask,
                       const Twine &Name = ""),
                  ARGS(V1, V2, Mask, Name));
  EMBED_THE_CHECK(CreateShuffleVector,
                  Value *,
                  ARGS(Value *V1,
                       Value *V2,
                       ArrayRef<int> Mask,
                       const Twine &Name = ""),
                  ARGS(V1, V2, Mask, Name));
  EMBED_THE_CHECK(CreateShuffleVector,
                  Value *,
                  ARGS(Value *V, ArrayRef<int> Mask, const Twine &Name = ""),
                  ARGS(V, Mask, Name));
  EMBED_THE_CHECK(CreateExtractValue,
                  Value *,
                  ARGS(Value *Agg,
                       ArrayRef<unsigned> Idxs,
                       const Twine &Name = ""),
                  ARGS(Agg, Idxs, Name));
  EMBED_THE_CHECK(CreateInsertValue,
                  Value *,
                  ARGS(Value *Agg,
                       Value *Val,
                       ArrayRef<unsigned> Idxs,
                       const Twine &Name = ""),
                  ARGS(Agg, Val, Idxs, Name));
  EMBED_THE_CHECK(CreateLandingPad,
                  LandingPadInst *,
                  ARGS(Type *Ty, unsigned NumClauses, const Twine &Name = ""),
                  ARGS(Ty, NumClauses, Name));
  EMBED_THE_CHECK(CreateFreeze,
                  Value *,
                  ARGS(Value *V, const Twine &Name = ""),
                  ARGS(V, Name));
  EMBED_THE_CHECK(CreateIsNull,
                  Value *,
                  ARGS(Value *Arg, const Twine &Name = ""),
                  ARGS(Arg, Name));
  EMBED_THE_CHECK(CreateIsNotNull,
                  Value *,
                  ARGS(Value *Arg, const Twine &Name = ""),
                  ARGS(Arg, Name));
  EMBED_THE_CHECK(CreateIsNeg,
                  Value *,
                  ARGS(Value *Arg, const Twine &Name = ""),
                  ARGS(Arg, Name));
  EMBED_THE_CHECK(CreateIsNotNeg,
                  Value *,
                  ARGS(Value *Arg, const Twine &Name = ""),
                  ARGS(Arg, Name));
  EMBED_THE_CHECK(CreatePtrDiff,
                  Value *,
                  ARGS(Type *ElemTy,
                       Value *LHS,
                       Value *RHS,
                       const Twine &Name = ""),
                  ARGS(ElemTy, LHS, RHS, Name));
  EMBED_THE_CHECK(CreateLaunderInvariantGroup,
                  Value *,
                  ARGS(Value *Ptr),
                  ARGS(Ptr));
  EMBED_THE_CHECK(CreateStripInvariantGroup,
                  Value *,
                  ARGS(Value *Ptr),
                  ARGS(Ptr));
  EMBED_THE_CHECK(CreateVectorReverse,
                  Value *,
                  ARGS(Value *V, const Twine &Name = ""),
                  ARGS(V, Name));
  EMBED_THE_CHECK(CreateVectorSplice,
                  Value *,
                  ARGS(Value *V1,
                       Value *V2,
                       int64_t Imm,
                       const Twine &Name = ""),
                  ARGS(V1, V2, Imm, Name));
  EMBED_THE_CHECK(CreateVectorSplat,
                  Value *,
                  ARGS(unsigned NumElts, Value *V, const Twine &Name = ""),
                  ARGS(NumElts, V, Name));
  EMBED_THE_CHECK(CreateVectorSplat,
                  Value *,
                  ARGS(ElementCount EC, Value *V, const Twine &Name = ""),
                  ARGS(EC, V, Name));
  EMBED_THE_CHECK(CreateExtractInteger,
                  Value *,
                  ARGS(const DataLayout &DL,
                       Value *From,
                       IntegerType *ExtractedTy,
                       uint64_t Offset,
                       const Twine &Name),
                  ARGS(DL, From, ExtractedTy, Offset, Name));
  EMBED_THE_CHECK(CreatePreserveArrayAccessIndex,
                  Value *,
                  ARGS(Type *ElTy,
                       Value *Base,
                       unsigned Dimension,
                       unsigned LastIndex,
                       MDNode *DbgInfo),
                  ARGS(ElTy, Base, Dimension, LastIndex, DbgInfo));
  EMBED_THE_CHECK(CreatePreserveUnionAccessIndex,
                  Value *,
                  ARGS(Value *Base, unsigned FieldIndex, MDNode *DbgInfo),
                  ARGS(Base, FieldIndex, DbgInfo));
  EMBED_THE_CHECK(CreatePreserveStructAccessIndex,
                  Value *,
                  ARGS(Type *ElTy,
                       Value *Base,
                       unsigned Index,
                       unsigned FieldIndex,
                       MDNode *DbgInfo),
                  ARGS(ElTy, Base, Index, FieldIndex, DbgInfo));
  EMBED_THE_CHECK(CreateAlignmentAssumption,
                  CallInst *,
                  ARGS(const DataLayout &DL,
                       Value *PtrValue,
                       unsigned Alignment,
                       Value *OffsetValue = nullptr),
                  ARGS(DL, PtrValue, Alignment, OffsetValue));
  EMBED_THE_CHECK(CreateAlignmentAssumption,
                  CallInst *,
                  ARGS(const DataLayout &DL,
                       Value *PtrValue,
                       Value *Alignment,
                       Value *OffsetValue = nullptr),
                  ARGS(DL, PtrValue, Alignment, OffsetValue));

#undef ARGS
#undef EMBED_THE_CHECK
#undef SKIP_THE_CHECK
};

} // namespace llvm

namespace revng {

/// This is a wrapper over llvm alternative that force-sets a debug location
/// even when it's the insertion point is a basic block.
///
/// On top of that, it asserts that *every* created instruction is created
/// with valid debug information attached.
class IRBuilder : public llvm::RevngIRBuilderWrapper {
  // Using `class` here instead of `using` to allow forward declaring
  // `class IRBuilder` elsewhere without including this bulky header.
public:
  using llvm::RevngIRBuilderWrapper::RevngIRBuilderWrapper;
};

} // namespace revng
