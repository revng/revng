#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <compare>
namespace llvm {
class Function;
class LLVMContext;
class Module;
class Type;
class ExtractValueInst;
} // end namespace llvm

#include "revng/Support/FunctionTags.h"
#include "revng/Support/OpaqueFunctionsPool.h"

/// This struct can be used as a key of an OpaqueFunctionsPool where both
/// the return type and one of the arguments are needed to identify a function
/// in the pool.
struct TypePair {
  llvm::Type *RetType;
  llvm::Type *ArgType;

  bool operator<(const TypePair &Rhs) const {
    return RetType < Rhs.RetType
           or (RetType == Rhs.RetType and ArgType < Rhs.ArgType);
  }
};

/// AddressOf functions are used to transform a reference into a pointer.
///
/// \param RetType The LLVM type returned by the Addressof call
/// \param BaseType The LLVM type of the second argument (the reference that
/// we want to transform into a pointer).
llvm::FunctionType *getAddressOfType(llvm::Type *RetType, llvm::Type *BaseType);

/// Initializes a pool of AddressOf functions, initializing it its internal
/// Module
void initAddressOfPool(OpaqueFunctionsPool<TypePair> &Pool, llvm::Module *M);

struct StringLiteralPoolKey {
  MetaAddress Address;
  uint64_t VirtualSize;
  uint64_t OffsetInSegment;
  llvm::Type *Type;

  std::strong_ordering
  operator<=>(const StringLiteralPoolKey &) const = default;
};

/// Initializes a pool of StringLiteral functions.
void initStringLiteralPool(OpaqueFunctionsPool<StringLiteralPoolKey> &Pool,
                           llvm::Module *M);

/// Initializes a pool of Parentheses functions
void initParenthesesPool(OpaqueFunctionsPool<llvm::Type *> &Pool);

/// Initializes a pool of hex literals printing functions
void initHexPrintPool(OpaqueFunctionsPool<llvm::Type *> &Pool);

/// Initializes a pool of char literals printing functions
void initCharPrintPool(OpaqueFunctionsPool<llvm::Type *> &Pool);

/// Initializes a pool of bool literals printing functions
void initBoolPrintPool(OpaqueFunctionsPool<llvm::Type *> &Pool);

/// Initializes a pool of NULL pointer literals printing functions
void initNullPtrPrintPool(OpaqueFunctionsPool<llvm::Type *> &Pool);

/// Initializes a pool of unary_minus functions
void initUnaryMinusPool(OpaqueFunctionsPool<llvm::Type *> &Pool);

/// Initializes a pool of binary_not functions
void initBinaryNotPool(OpaqueFunctionsPool<llvm::Type *> &Pool);

/// Initializes a pool of boolean_not functions
void initBooleanNotPool(OpaqueFunctionsPool<llvm::Type *> &Pool);

/// ModelGEP functions are used to replace pointer arithmetic with a navigation
/// of the Model.
///
/// \param RetType ModelGEP should return an integer of the size of the gepped
/// element
/// \param BaseType The LLVM type of the second argument (the base pointer)
llvm::Function *
getModelGEP(llvm::Module &M, llvm::Type *RetType, llvm::Type *BaseType);

/// ModelGEP Ref is a ModelGEP where the base value is considered to be a
/// reference.
llvm::Function *
getModelGEPRef(llvm::Module &M, llvm::Type *RetType, llvm::Type *BaseType);

using ModelCastPoolKey = std::pair<llvm::Type *, llvm::Type *>;

/// Initializes a pool of ModelCast functions
void initModelCastPool(OpaqueFunctionsPool<TypePair> &Pool, llvm::Module *M);

using SegmentRefPoolKey = std::tuple<MetaAddress, uint64_t, llvm::Type *>;

/// Initializes a pool of SegmentRef functions
void initSegmentRefPool(OpaqueFunctionsPool<SegmentRefPoolKey> &Pool,
                        llvm::Module *M);

/// Derive the function type of the corresponding OpaqueExtractValue() function
/// from an ExtractValue instruction. OpaqueExtractValues wrap an
/// ExtractValue to prevent it from being optimized out, so the return type and
/// arguments are the same as the instruction being wrapped.
llvm::FunctionType *getOpaqueEVFunctionType(llvm::ExtractValueInst *Extract);

// Initializes a pool of OpaqueExtractValue instructions, so that a new one can
// be created on-demand.
void initOpaqueEVPool(OpaqueFunctionsPool<TypePair> &Pool, llvm::Module *M);

/// LocalVariable is used to indicate the allocation of a local variable. It
/// returns a reference to the allocated variable.
llvm::FunctionType *getLocalVarType(llvm::Type *ReturnedType);

/// Initializes a pool of LocalVariable functions, initializing it its internal
/// Module.
void initLocalVarPool(OpaqueFunctionsPool<llvm::Type *> &Pool);

/// Assign() are meant to replace `store` instructions in which the pointer
/// operand is a reference.
llvm::FunctionType *getAssignFunctionType(llvm::Type *ValueType,
                                          llvm::Type *PtrType);

/// Initializes a pool of Assign functions, initializing it its internal
/// Module.
void initAssignPool(OpaqueFunctionsPool<llvm::Type *> &Pool);

/// Copy() are meant to replace `load` instructions in which the pointer
/// operand is a reference.
llvm::FunctionType *getCopyType(llvm::Type *ReturnedType);

/// Initializes a pool of Copy functions, initializing it its internal
/// Module.
void initCopyPool(OpaqueFunctionsPool<llvm::Type *> &Pool);
