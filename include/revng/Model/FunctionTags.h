#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Support/IRBuilder.h"
#include "revng/Support/IRHelperRegistry.h"
#include "revng/Support/OpaqueFunctionsPool.h"

std::tuple<MetaAddress, uint64_t, uint64_t, uint64_t, llvm::Type *>
extractStringLiteralFromMetadata(const llvm::Function &StringLiteralFunction);

/// Extract the key of a model::Segment stored as a metadata.
std::pair<MetaAddress, uint64_t>
extractSegmentKeyFromMetadata(const llvm::Function &F);

namespace FunctionTags {

extern Tag QEMU;
extern Tag Helper;
extern Tag ABIEnforced;
extern Tag CSVsPromoted;
extern Tag Exceptional;
extern Tag StructInitializer;
extern Tag OpaqueCSVValue;
extern Tag FunctionDispatcher;
extern Tag Root;
extern Tag IsolatedRoot;
extern Tag CSVsAsArgumentsWrapper;
extern Tag Marker;
extern Tag DynamicFunction;
extern Tag ClobbererFunction;
extern Tag WriterFunction;
extern Tag ReaderFunction;
extern Tag OpaqueReturnAddressFunction;
extern Tag CSV;
inline const char *UniqueIDMDName = "revng.unique_id";
extern Tag AllocatesLocalVariable;
extern Tag ReturnsPolymorphic;
extern Tag IsRef;
extern Tag ScopeCloserMarker;
extern Tag GotoBlockMarker;

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

extern FunctionPoolTag<TypePair> AddressOf;

struct StringLiteralPoolKey {
  MetaAddress Address;
  uint64_t VirtualSize;
  uint64_t OffsetInSegment;
  llvm::Type *Type;

  std::strong_ordering
  operator<=>(const StringLiteralPoolKey &) const = default;
};

extern FunctionPoolTag<StringLiteralPoolKey> StringLiteral;
extern FunctionPoolTag<TypePair> ModelCast;
extern Tag ModelGEP;
extern Tag ModelGEPRef;
extern FunctionPoolTag<TypePair> OpaqueExtractValue;
extern FunctionPoolTag<llvm::Type *> Parentheses;
extern Tag LiteralPrintDecorator;
extern FunctionPoolTag<llvm::Type *> HexInteger;
extern FunctionPoolTag<llvm::Type *> CharInteger;
extern FunctionPoolTag<llvm::Type *> BoolInteger;
extern FunctionPoolTag<llvm::Type *> NullPtr;
extern FunctionPoolTag<llvm::Type *> LocalVariable;
extern FunctionPoolTag<llvm::Type *> Assign;
extern FunctionPoolTag<llvm::Type *> Copy;
using SegmentRefPoolKey = std::tuple<MetaAddress, uint64_t, llvm::Type *>;
extern FunctionPoolTag<SegmentRefPoolKey> SegmentRef;
extern FunctionPoolTag<llvm::Type *> UnaryMinus;
extern FunctionPoolTag<llvm::Type *> BinaryNot;
extern FunctionPoolTag<llvm::Type *> BooleanNot;
extern Tag LiftingArtifactsRemoved;
extern Tag StackPointerPromoted;
extern Tag StackAccessesSegregated;
extern Tag Decompiled;
extern Tag StackOffsetMarker;
extern Tag BinaryOperationOverflows;
extern Tag Comment;

} // namespace FunctionTags

inline bool isRootOrLifted(const llvm::Function *F) {
  auto Tags = FunctionTags::TagsSet::from(F);
  return Tags.contains(FunctionTags::Root)
         or Tags.contains(FunctionTags::Isolated);
}

inline const llvm::CallInst *getCallToHelper(const llvm::Instruction *I) {
  revng_assert(I != nullptr);
  const llvm::Function *Callee = getCallee(I);
  if (Callee != nullptr && FunctionTags::Helper.isTagOf(Callee))
    return llvm::cast<llvm::CallInst>(I);
  else
    return nullptr;
}

inline llvm::CallInst *getCallToHelper(llvm::Instruction *I) {
  revng_assert(I != nullptr);
  const llvm::Function *Callee = getCallee(I);
  if (Callee != nullptr && FunctionTags::Helper.isTagOf(Callee))
    return llvm::cast<llvm::CallInst>(I);
  else
    return nullptr;
}

/// Is \p I a call to an helper function?
inline bool isCallToHelper(const llvm::Instruction *I) {
  return getCallToHelper(I) != nullptr;
}

std::optional<CSVsUsage> tryGetCSVUsedByHelperCall(llvm::Instruction *Call);

inline CSVsUsage getCSVUsedByHelperCall(llvm::Instruction *Call) {
  return tryGetCSVUsedByHelperCall(Call).value();
}

/// Checks if \p I is a marker
///
/// A marker a function call to an empty function acting as meta-information,
/// for example the `function_call` marker.
inline bool isMarker(const llvm::Instruction *I) {
  if (auto *Callee = getCallee(I))
    return FunctionTags::Marker.isTagOf(Callee);

  return false;
}

inline llvm::Instruction *nextNonMarker(llvm::Instruction *I) {
  auto It = I->getIterator();
  auto End = I->getParent()->end();
  while (++It != End) {
    if (not isMarker(&*It))
      return &*It;
  }

  return nullptr;
}

inline llvm::CallInst *getMarker(llvm::Instruction *T, llvm::Function *Marker) {
  revng_assert(T && T->isTerminator());
  llvm::Instruction *Previous = getPrevious(T);
  while (Previous != nullptr
         && (isMarker(Previous) || isCallTo(Previous, "abort"))) {
    if (auto *Call = getCallTo(Previous, Marker))
      return Call;

    Previous = getPrevious(Previous);
  }

  return nullptr;
}

inline llvm::CallInst *getMarker(llvm::Instruction *I,
                                 llvm::StringRef MarkerName) {
  return getMarker(I, getIRHelper(MarkerName, *getModule(I)));
}

inline llvm::CallInst *getMarker(llvm::BasicBlock *BB,
                                 llvm::StringRef MarkerName) {
  return getMarker(BB->getTerminator(), MarkerName);
}

inline llvm::CallInst *getMarker(llvm::BasicBlock *BB, llvm::Function *Marker) {
  return getMarker(BB->getTerminator(), Marker);
}

inline bool hasMarker(llvm::Instruction *T, llvm::StringRef MarkerName) {
  return getMarker(T, MarkerName);
}

inline bool hasMarker(llvm::BasicBlock *BB, llvm::StringRef MarkerName) {
  return getMarker(BB->getTerminator(), MarkerName);
}

inline bool hasMarker(llvm::Instruction *T, llvm::Function *Marker) {
  return getMarker(T, Marker);
}

inline bool hasMarker(llvm::BasicBlock *BB, llvm::Function *Marker) {
  return getMarker(BB->getTerminator(), Marker);
}

/// Return the callee basic block given a function_call marker.
inline llvm::BasicBlock *getFunctionCallCallee(llvm::Instruction *T) {
  if (auto *Call = getMarker(T, "function_call")) {
    if (auto *Callee = llvm::dyn_cast<llvm::BlockAddress>(Call->getOperand(0)))
      return Callee->getBasicBlock();
  }

  return nullptr;
}

inline llvm::BasicBlock *getFunctionCallCallee(llvm::BasicBlock *BB) {
  return getFunctionCallCallee(BB->getTerminator());
}

/// Return the fall-through basic block given a function_call marker.
inline llvm::BasicBlock *getFallthrough(llvm::Instruction *T) {
  if (auto *Call = getMarker(T, "function_call")) {
    auto *Fallthrough = llvm::cast<llvm::BlockAddress>(Call->getOperand(1));
    return Fallthrough->getBasicBlock();
  }

  return nullptr;
}

inline llvm::BasicBlock *getFallthrough(llvm::BasicBlock *BB) {
  return getFallthrough(BB->getTerminator());
}

/// Return true if \p T is has a fallthrough basic block.
inline bool isFallthrough(llvm::Instruction *T) {
  return getFallthrough(T) != nullptr;
}

inline bool isFallthrough(llvm::BasicBlock *BB) {
  return isFallthrough(BB->getTerminator());
}

llvm::SmallVector<llvm::SmallPtrSet<llvm::CallInst *, 2>, 2>
getExtractedValuesFromInstruction(llvm::Instruction *);

llvm::SmallVector<llvm::SmallPtrSet<const llvm::CallInst *, 2>, 2>
getExtractedValuesFromInstruction(const llvm::Instruction *);

/// Set the key of a model::Segment stored as a metadata.
void setSegmentKeyMetadata(llvm::Function &SegmentRefFunction,
                           MetaAddress StartAddress,
                           uint64_t VirtualSize);

/// Returns true if \F has an attached metadata representing a segment key.
bool hasSegmentKeyMetadata(const llvm::Function &F);

void setStringLiteralMetadata(llvm::Function &StringLiteralFunction,
                              MetaAddress StartAddress,
                              uint64_t VirtualSize,
                              uint64_t Offset,
                              uint64_t StringLength,
                              llvm::Type *ReturnType);

bool hasStringLiteralMetadata(const llvm::Function &StringLiteralFunction);

inline constexpr llvm::StringRef AbortFunctionName = "revng_abort";

/// \p PCH if not nullptr, the function will force the program counter CSVs to
///    a sensible value for better debugging.
llvm::CallInst &emitAbort(revng::IRBuilder &Builder,
                          const llvm::Twine &Message,
                          const llvm::DebugLoc &DbgLocation = {},
                          const ProgramCounterHandler *PCH = nullptr);

inline llvm::CallInst &emitAbort(llvm::Instruction *InsertionPoint,
                                 const llvm::Twine &Message,
                                 const llvm::DebugLoc &DbgLocation = {},
                                 const ProgramCounterHandler *PCH = nullptr) {
  revng::NonDebugInfoCheckingIRBuilder Builder(InsertionPoint);
  return emitAbort(Builder, Message, DbgLocation, PCH);
}

inline llvm::CallInst &emitAbort(llvm::BasicBlock *InsertionPoint,
                                 const llvm::Twine &Message,
                                 const llvm::DebugLoc &DbgLocation = {},
                                 const ProgramCounterHandler *PCH = nullptr) {
  revng::NonDebugInfoCheckingIRBuilder Builder(InsertionPoint, DbgLocation);
  return emitAbort(Builder, Message, DbgLocation, PCH);
}

/// \p PCH if not nullptr, the function will force the program counter CSVs to
///    a sensible value for better debugging.
llvm::CallInst &emitMessage(revng::IRBuilder &Builder,
                            const llvm::Twine &Message,
                            const llvm::DebugLoc &DbgLocation = {},
                            const ProgramCounterHandler *PCH = nullptr);

template<typename IPType>
  requires std::constructible_from<revng::IRBuilder, IPType>
llvm::CallInst &emitMessage(IPType &&InsertionPoint,
                            const llvm::Twine &Message,
                            const llvm::DebugLoc &DbgLocation = {},
                            const ProgramCounterHandler *PCH = nullptr) {
  revng::NonDebugInfoCheckingIRBuilder
    Builder(std::forward<IPType>(InsertionPoint));
  return emitMessage(Builder, Message, DbgLocation, PCH);
}

inline MetaAddress getMetaAddressOfIsolatedFunction(const llvm::Function &F) {
  revng_assert(FunctionTags::Isolated.isTagOf(&F));
  return getMetaAddressMetadata(&F, FunctionEntryMDName);
}

/// AddressOf functions are used to transform a reference into a pointer.
///
/// \param RetType The LLVM type returned by the Addressof call
/// \param BaseType The LLVM type of the second argument (the reference that
/// we want to transform into a pointer).
llvm::FunctionType *getAddressOfType(llvm::Type *RetType, llvm::Type *BaseType);

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

/// Derive the function type of the corresponding OpaqueExtractValue() function
/// from an ExtractValue instruction. OpaqueExtractValues wrap an
/// ExtractValue to prevent it from being optimized out, so the return type and
/// arguments are the same as the instruction being wrapped.
llvm::FunctionType *getOpaqueEVFunctionType(llvm::ExtractValueInst *Extract);

/// LocalVariable is used to indicate the allocation of a local variable. It
/// returns a reference to the allocated variable.
llvm::FunctionType *getLocalVarType(llvm::Type *ReturnedType);

/// Assign() are meant to replace `store` instructions in which the pointer
/// operand is a reference.
llvm::FunctionType *getAssignFunctionType(llvm::Type *ValueType,
                                          llvm::Type *PtrType);

/// Copy() are meant to replace `load` instructions in which the pointer
/// operand is a reference.
llvm::FunctionType *getCopyType(llvm::Type *ReturnedType,
                                llvm::Type *VariableReferenceType);

//
// {is,get}CallToIsolatedFunction
//
const llvm::CallInst *getCallToIsolatedFunction(const llvm::Value *V);

llvm::CallInst *getCallToIsolatedFunction(llvm::Value *V);

inline bool isCallToIsolatedFunction(const llvm::Value *V) {
  return getCallToIsolatedFunction(V) != nullptr;
}
