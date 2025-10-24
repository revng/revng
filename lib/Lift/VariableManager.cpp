/// \file VariableManager.cpp
/// This file handles the creation and management of global variables, i.e.
/// mainly parts of the CPU state

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <cstdint>
#include <set>
#include <sstream>
#include <stack>
#include <string>

#include "qemu/libtcg/libtcg.h"

#include "llvm/ADT/StringExtras.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/MDBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/Casting.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/ValueMapper.h"

#include "revng/Lift/VariableManager.h"
#include "revng/Model/FunctionTags.h"
#include "revng/Model/Register.h"
#include "revng/Support/Assert.h"
#include "revng/Support/CommandLine.h"
#include "revng/Support/Debug.h"
#include "revng/Support/IRHelpers.h"

// This name corresponds to a function in `early-linked`.
RegisterIRHelper SetRegisterMarker("set_register");

static Logger<> Log("csv-at-offset");

using namespace llvm;

class OffsetValueStack {

private:
  using OffsetValuePair = std::pair<int64_t, Value *>;

public:
  void pushIfNew(int64_t Offset, Value *V) {
    OffsetValuePair Element = { Offset, V };
    if (!Seen.contains(Element)) {
      Seen.insert(Element);
      Stack.push_back(Element);
    }
  }

  void push(int64_t Offset, Value *V) {
    OffsetValuePair Element = { Offset, V };
    Stack.push_back(Element);
  }

  bool empty() { return Stack.empty(); }

  std::pair<int64_t, Value *> pop() {
    auto Result = Stack.back();
    Stack.pop_back();
    return Result;
  }

  // TODO: this is on O(n)
  void cloneSisters(Value *Old, Value *New) {
    for (auto &OVP : Stack)
      if (OVP.second == Old)
        push(OVP.first, New);
  }

private:
  std::set<OffsetValuePair> Seen;
  std::vector<OffsetValuePair> Stack;
};

static std::pair<IntegerType *, unsigned>
getTypeAtOffset(const DataLayout *TheLayout, Type *VarType, intptr_t Offset) {
  std::string Prefix = "";
  while (true) {
    switch (VarType->getTypeID()) {
    case llvm::Type::TypeID::PointerTyID:
      // Ignore pointers
      revng_log(Log, Prefix << "Found a pointer. Bailing out.");
      return { nullptr, 0 };

    case llvm::Type::TypeID::IntegerTyID:
      revng_log(Log, Prefix << "Found a i" << VarType->getIntegerBitWidth());
      return { cast<IntegerType>(VarType), Offset };

    case llvm::Type::TypeID::ArrayTyID: {
      auto ElementsCount = VarType->getArrayNumElements();
      VarType = VarType->getArrayElementType();
      auto TypeSize = TheLayout->getTypeAllocSize(VarType);
      auto Index = Offset / TypeSize;
      Offset %= TypeSize;
      revng_log(Log,
                Prefix << "Element " << Index << " in an array of "
                       << ElementsCount
                       << " elements. Offset in Element: " << Offset);
    } break;

    case llvm::Type::TypeID::StructTyID: {
      StructType *TheStruct = cast<StructType>(VarType);
      const StructLayout *Layout = TheLayout->getStructLayout(TheStruct);
      unsigned FieldIndex = Layout->getElementContainingOffset(Offset);
      uint64_t FieldOffset = Layout->getElementOffset(FieldIndex);
      VarType = TheStruct->getTypeAtIndex(FieldIndex);
      intptr_t FieldEnd = FieldOffset + TheLayout->getTypeAllocSize(VarType);

      revng_log(Log,
                Prefix << "Offset: " << Offset
                       << "; Struct Name: " << TheStruct->getName().str()
                       << "; Field Index: " << FieldIndex << "; Field offset: "
                       << FieldOffset << "; Field end: " << FieldEnd << ".");

      if (Offset >= FieldEnd) {
        revng_log(Log, Prefix << "It's padding. Bailing out.");
        return { nullptr, 0 };
      }

      Offset -= FieldOffset;
    } break;

    default:
      revng_abort("Unexpected TypeID");
    }

    Prefix += "  ";
  }
}

VariableManager::VariableManager(Module &M,
                                 bool TargetIsLittleEndian,
                                 unsigned LibTcgEnvOffset,
                                 uint8_t *LibTcgEnvPtr,
                                 const std::map<intptr_t, StringRef>
                                   &GlobalNames) :
  TheModule(M),
  AllocaBuilder(getContext(&M)),
  ArchCPUStruct(nullptr),
  ModuleLayout(&TheModule.getDataLayout()),
  LibTcgEnvOffset(LibTcgEnvOffset),
  LibTcgEnvPtr(LibTcgEnvPtr),
  Env(nullptr),
  TargetIsLittleEndian(TargetIsLittleEndian),
  GlobalNames(GlobalNames) {

  // Reminder:
  //
  // struct ArchCPU {
  //   CPUState parent_obj;
  //   CPUX86State /* aka CPUArchState */ env;
  // };

  // TODO: this not very robust. We should have a function with a sensible name
  //       taking as argument ${ARCH}CPU so that we can easily identify the
  //       struct.
  ArchCPUStruct = StructType::getTypeByName(M.getContext(), "struct.ArchCPU");
  revng_assert(ArchCPUStruct != nullptr);

  revng_assert(LibTcgEnvPtr != nullptr);

  IntegerType *IntPtrTy = AllocaBuilder.getIntPtrTy(*ModuleLayout);
  Env = cast<GlobalVariable>(TheModule.getOrInsertGlobal("env", IntPtrTy));
  Env->setInitializer(ConstantInt::get(IntPtrTy, LibTcgEnvOffset));
}

std::optional<StoreInst *>
VariableManager::storeToCPUStateOffset(revng::IRBuilder &Builder,
                                       unsigned StoreSize,
                                       unsigned Offset,
                                       Value *ToStore) {
  GlobalVariable *Target = nullptr;
  unsigned Remaining;
  std::tie(Target, Remaining) = getByCPUStateOffsetWithRemainder(Offset);

  if (Target == nullptr)
    return {};

  unsigned ShiftAmount = 0;
  if (TargetIsLittleEndian)
    ShiftAmount = Remaining;
  else {
    // >> (Size1 - Size2) - Remaining;
    Type *PointeeTy = Target->getValueType();
    unsigned GlobalSize = cast<IntegerType>(PointeeTy)->getBitWidth() / 8;
    revng_assert(GlobalSize != 0);
    ShiftAmount = (GlobalSize - StoreSize) - Remaining;
  }
  ShiftAmount *= 8;

  // Build blanking mask
  uint64_t BitMask = (StoreSize == 8 ? (uint64_t) -1 :
                                       ((uint64_t) 1 << StoreSize * 8) - 1);
  revng_assert(ShiftAmount != 64);
  BitMask <<= ShiftAmount;
  BitMask = ~BitMask;

  auto *InputStoreTy = cast<IntegerType>(Builder.getIntNTy(StoreSize * 8));
  auto *FieldTy = cast<IntegerType>(Target->getValueType());
  unsigned FieldSize = FieldTy->getBitWidth() / 8;

  // Are we trying to store more than it fits?
  if (StoreSize > FieldSize) {
    // If we're storing more than it fits and the following memory is not
    // padding the store is not valid.
    if (getByCPUStateOffsetWithRemainder(Offset + FieldSize).first != nullptr)
      return {};
  }

  // Truncate value to store
  auto *Truncated = Builder.CreateTrunc(ToStore, InputStoreTy);
  if (StoreSize > FieldSize)
    Truncated = Builder.CreateTrunc(Truncated, FieldTy);

  // Re-extend
  ToStore = Builder.CreateZExt(Truncated, FieldTy);

  if (BitMask != 0 and StoreSize != FieldSize) {
    // Load the value
    auto *LoadEnvField = createLoad(Builder, Target);

    auto *Blanked = Builder.CreateAnd(LoadEnvField, BitMask);

    // Shift value to store
    ToStore = Builder.CreateShl(ToStore, ShiftAmount);

    // Combine them
    ToStore = Builder.CreateOr(ToStore, Blanked);
  }

  return { Builder.CreateStore(ToStore, Target) };
}

Value *VariableManager::loadFromCPUStateOffset(revng::IRBuilder &Builder,
                                               unsigned LoadSize,
                                               unsigned Offset) {
  GlobalVariable *Target = nullptr;
  unsigned Remaining;
  std::tie(Target, Remaining) = getByCPUStateOffsetWithRemainder(Offset);

  if (Target == nullptr)
    return nullptr;

  // Load the whole field
  auto *LoadEnvField = createLoad(Builder, Target);

  // Extract the desired part
  // Shift right of the desired amount
  unsigned ShiftAmount = 0;
  if (TargetIsLittleEndian) {
    ShiftAmount = Remaining;
  } else {
    // >> (Size1 - Size2) - Remaining;
    auto *LoadedTy = cast<IntegerType>(LoadEnvField->getType());
    unsigned GlobalSize = LoadedTy->getBitWidth() / 8;
    revng_assert(GlobalSize != 0);
    ShiftAmount = (GlobalSize - LoadSize) - Remaining;
  }
  ShiftAmount *= 8;
  Value *Result = LoadEnvField;

  if (ShiftAmount != 0)
    Result = Builder.CreateLShr(Result, ShiftAmount);

  Type *LoadTy = Builder.getIntNTy(LoadSize * 8);

  // Are we trying to load more than its available in the field?
  if (auto FieldTy = dyn_cast<IntegerType>(Result->getType())) {
    unsigned FieldSize = FieldTy->getBitWidth() / 8;
    if (FieldSize < LoadSize) {
      // If after what we are loading there is something that is not padding we
      // cannot load safely
      if (getByCPUStateOffsetWithRemainder(Offset + FieldSize).first != nullptr)
        return nullptr;
      Result = Builder.CreateZExt(Result, LoadTy);
    }
  }

  // Truncate of the desired amount
  return Builder.CreateTrunc(Result, LoadTy);
}

void VariableManager::memOpAtEnvOffset(revng::IRBuilder &Builder,
                                       llvm::CallInst *Call,
                                       unsigned InitialEnvOffset,
                                       bool EnvIsSrc) {
  Function *Callee = getCallee(Call);
  // We only support memcpys where the last parameter is constant
  revng_assert(Callee != nullptr);
  bool IsMemset = Callee->getIntrinsicID() == Intrinsic::memset;
  revng_assert(Callee->getIntrinsicID() == Intrinsic::memcpy
               or Callee->getIntrinsicID() == Intrinsic::memmove or IsMemset);
  revng_assert(isa<ConstantInt>(Call->getArgOperand(2)));

  Value *OtherOp = Call->getArgOperand(EnvIsSrc ? 0 : 1);
  auto *MemcpySize = cast<Constant>(Call->getArgOperand(2));
  Value *OtherBasePtr = nullptr;

  if (not IsMemset)
    OtherBasePtr = Builder.CreatePtrToInt(OtherOp, Builder.getInt64Ty());

  uint64_t TotalSize = getZExtValue(MemcpySize, *ModuleLayout);
  uint64_t Offset = 0;

  bool OnlyPointersAndPadding = true;
  while (Offset < TotalSize) {
    GlobalVariable *EnvVar = getByEnvOffset(InitialEnvOffset + Offset).first;

    // Consider the case when there's simply nothing there (alignment space).
    if (EnvVar == nullptr) {
      Offset++;
      continue;
    }
    OnlyPointersAndPadding = false;

    ConstantInt *OffsetInt = Builder.getInt64(Offset);
    Value *NewAddress = nullptr;
    Value *OtherPtr = nullptr;

    if (not IsMemset) {
      NewAddress = Builder.CreateAdd(OffsetInt, OtherBasePtr);
      OtherPtr = Builder.CreateIntToPtr(NewAddress, EnvVar->getType());
    }

    StoreInst *New = nullptr;
    if (EnvIsSrc) {
      revng_assert(not IsMemset);
      New = Builder.CreateStore(createLoad(Builder, EnvVar), OtherPtr);
    } else {
      Type *CSVType = EnvVar->getValueType();
      Value *ToStore = nullptr;
      if (IsMemset) {
        // TODO: handle non-zero memset
        Value *SetValue = Call->getArgOperand(1);
        revng_assert(cast<ConstantInt>(SetValue)->getValue().isZero());
        ToStore = ConstantInt::get(CSVType, 0);
      } else {
        ToStore = Builder.CreateLoad(EnvVar->getValueType(), OtherPtr);
      }
      New = Builder.CreateStore(ToStore, EnvVar);
    }

    if (auto *GV = dyn_cast<GlobalVariable>(New->getPointerOperand())) {
      revng_assert(New->getValueOperand()->getType() == GV->getValueType());
    }

    Type *PointeeTy = EnvVar->getValueType();
    Offset += ModuleLayout->getTypeAllocSize(PointeeTy);
  }

  if (OnlyPointersAndPadding)
    eraseFromParent(cast<Instruction>(OtherBasePtr));

  revng_assert(Offset == TotalSize);
}

void VariableManager::finalize() {
  LLVMContext &Context = getContext(&TheModule);

  revng::NonDebugInfoCheckingIRBuilder Builder(Context);

  // Create the setRegister function
  auto *SetRegisterTy = FunctionType::get(Builder.getVoidTy(),
                                          { Builder.getInt32Ty(),
                                            Builder.getInt64Ty() },
                                          false);
  FunctionCallee SetRegisterC = getOrInsertIRHelper("set_register",
                                                    TheModule,
                                                    SetRegisterTy);
  auto *SetRegister = cast<Function>(SetRegisterC.getCallee());
  SetRegister->setLinkage(GlobalValue::ExternalLinkage);

  // Collect arguments
  auto ArgIt = SetRegister->arg_begin();
  auto ArgEnd = SetRegister->arg_end();
  revng_assert(ArgIt != ArgEnd);
  Argument *RegisterID = &*ArgIt;
  ArgIt++;
  revng_assert(ArgIt != ArgEnd);
  Argument *NewValue = &*ArgIt;
  ArgIt++;
  revng_assert(ArgIt == ArgEnd);

  // Create main basic blocks
  using BasicBlock = BasicBlock;
  auto *EntryBB = BasicBlock::Create(Context, "", SetRegister);
  auto *DefaultBB = BasicBlock::Create(Context, "", SetRegister);
  auto *ReturnBB = BasicBlock::Create(Context, "", SetRegister);

  // Populate the default case of the switch
  Builder.SetInsertPoint(DefaultBB);
  emitAbort(Builder, "");

  // Create the switch statement
  Builder.SetInsertPoint(EntryBB);
  auto *Switch = Builder.CreateSwitch(RegisterID,
                                      DefaultBB,
                                      CPUStateGlobals.size());
  for (auto &P : CPUStateGlobals) {
    auto *CSVIntTy = cast<IntegerType>(P.second->getValueType());
    if (CSVIntTy->getBitWidth() <= 64) {
      // Set the value of the CSV
      auto *SetRegisterBB = BasicBlock::Create(Context, "", SetRegister);
      Builder.SetInsertPoint(SetRegisterBB);
      Builder.CreateStore(Builder.CreateTrunc(NewValue, CSVIntTy), P.second);
      Builder.CreateBr(ReturnBB);

      // Add the case to the switch
      Switch->addCase(Builder.getInt32(P.first), SetRegisterBB);
    }
  }

  // Finally, populate the return basic block
  Builder.SetInsertPoint(ReturnBB);
  Builder.CreateRetVoid();
}

bool VariableManager::isEnv(Value *TheValue) {
  auto *Load = dyn_cast<LoadInst>(TheValue);
  if (Load != nullptr)
    return Load->getPointerOperand() == Env;

  return TheValue == Env;
}

static ConstantInt *fromBytes(IntegerType *Type, void *Data) {
  switch (Type->getBitWidth()) {
  case 8:
    return ConstantInt::get(Type, *(static_cast<uint8_t *>(Data)));
  case 16:
    return ConstantInt::get(Type, *(static_cast<uint16_t *>(Data)));
  case 32:
    return ConstantInt::get(Type, *(static_cast<uint32_t *>(Data)));
  case 64:
    return ConstantInt::get(Type, *(static_cast<uint64_t *>(Data)));
  }

  revng_unreachable("Unexpected type");
}

// TODO: document that it can return nullptr
GlobalVariable *VariableManager::getByCPUStateOffset(intptr_t Offset) {
  GlobalVariable *Result = nullptr;
  unsigned Remaining;
  std::tie(Result, Remaining) = getByCPUStateOffsetWithRemainder(Offset);
  revng_assert(Remaining == 0);
  return Result;
}

std::optional<std::pair<GlobalVariable *, unsigned>>
VariableManager::getGlobalByCPUStateOffset(intptr_t Offset) const {
  auto It = CPUStateGlobals.upper_bound(Offset);

  // If we're earlier than the first one, bail out
  if (It == CPUStateGlobals.begin())
    return std::nullopt;

  // Move back of one position
  --It;

  // Compute GlobalVariable size
  intptr_t Size = It->second->getValueType()->getIntegerBitWidth() / 8;
  revng_assert(Size != 0);

  // Check if we're within the variable
  if (It->first <= Offset and Offset < (It->first + Size)) {
    // Return the global + offset within it
    return { { It->second, Offset - It->first } };
  }

  return std::nullopt;
}

std::pair<GlobalVariable *, unsigned>
VariableManager::getByCPUStateOffsetWithRemainder(intptr_t Offset) {

  // Check if we already created a variable for this offset
  if (auto MaybeResult = getGlobalByCPUStateOffset(Offset))
    return MaybeResult.value();

  revng_log(Log, "Considering offset " << Offset);
  LoggerIndent<> Indent(Log);

  // Get the type of the field at that offset (if any) and obtain the offset
  // within the field
  auto [VariableType,
        Remaining] = getTypeAtOffset(ModuleLayout, ArchCPUStruct, Offset);

  // Unsupported type, let the caller handle the situation
  if (VariableType == nullptr) {
    revng_log(Log, "Unsupported bailing out.");
    return { nullptr, 0 };
  }

  // Compute the actual start offset (discarding the offset within the global)
  auto GlobalOffset = Offset - Remaining;
  revng_assert(not CPUStateGlobals.contains(GlobalOffset));

  // Compute the name
  auto NameIt = GlobalNames.find(GlobalOffset);
  std::string Name;
  if (NameIt != GlobalNames.end()) {
    Name = "_" + NameIt->second.str();
  } else {
    static const char *UnknownCSVPrefix = "_state_0x";
    Name = UnknownCSVPrefix + utohexstr(GlobalOffset, true);
  }
  revng_log(Log, "Name " << Name);

  // TODO: if this is CSV, check it's of the correct size we expect

  // Check if a previous VariableManager has already created this variable
  if (auto *Result = TheModule.getGlobalVariable(Name, true)) {
    revng_log(Log, "It already exists.");
    // Check the variable looks like what we'd create
    revng_assert(Result->getValueType() == VariableType);
    revng_assert(FunctionTags::CSV.isTagOf(Result));
    revng_assert(Result->hasInitializer());
    revng_assert(not Result->isConstant());
    revng_assert(Result->getLinkage() == GlobalValue::ExternalLinkage);

    // Record and return it
    CPUStateGlobals[GlobalOffset] = Result;
    return { Result, Remaining };
  }

  revng_log(Log, "Creating.");

  auto InitializerPointer = LibTcgEnvPtr - LibTcgEnvOffset + GlobalOffset;
  revng_assert(InitializerPointer >= LibTcgEnvPtr - LibTcgEnvOffset);
  auto *InitialValue = fromBytes(cast<IntegerType>(VariableType),
                                 InitializerPointer);

  // Create the global
  auto *NewVariable = new GlobalVariable(TheModule,
                                         VariableType,
                                         false,
                                         GlobalValue::ExternalLinkage,
                                         InitialValue,
                                         Name);
  FunctionTags::CSV.addTo(NewVariable);

  // Register the variable
  CPUStateGlobals[GlobalOffset] = NewVariable;

  return { NewVariable, Remaining };
}

std::pair<bool, Value *> VariableManager::getOrCreate(LibTcgArgument *Argument,
                                                      bool Reading) {
  Type *VariableType = Argument->temp->type == LIBTCG_TYPE_I32 ?
                         AllocaBuilder.getInt32Ty() :
                         AllocaBuilder.getInt64Ty();

  switch (Argument->kind) {
  case LIBTCG_ARG_TEMP:
    switch (Argument->temp->kind) {
    case LIBTCG_TEMP_EBB: {
      // Temporary is dead at the end of the Extended Basic Block (EBB), the
      // single entry, multiple exit region that falls through basic blocks.
      auto It = EBBTemporaries.find(Argument->temp);
      if (It != EBBTemporaries.end()) {
        return { false, It->second };
      } else {
        // Can't read a temporary if it has never been written, we're probably
        // translating rubbish
        if (Reading)
          return { false, nullptr };

        AllocaInst *NewTemporary = AllocaBuilder.CreateAlloca(VariableType);
        EBBTemporaries[Argument->temp] = NewTemporary;
        return { true, NewTemporary };
      }
    }
    case LIBTCG_TEMP_TB: {
      // Temporary is dead at the end of the Translation Block (TB)
      auto It = TBTemporaries.find(Argument->temp);
      if (It != TBTemporaries.end()) {
        return { false, It->second };
      } else {
        AllocaInst *NewTemporary = AllocaBuilder.CreateAlloca(VariableType);
        TBTemporaries[Argument->temp] = NewTemporary;
        return { true, NewTemporary };
      }
    }
    case LIBTCG_TEMP_GLOBAL: {
      // Temporary is alive at the end of a Translation Block (TB), and
      // in between TBs
      Value *Result = getByCPUStateOffset(LibTcgEnvOffset
                                          + Argument->temp->mem_offset);
      revng_assert(Result != nullptr);
      return { false, Result };
    }
    case LIBTCG_TEMP_FIXED: {
      revng_assert(std::string(Argument->temp->name) == "env");
      revng_assert(Env != nullptr);
      return { false, Env };
    }
    case LIBTCG_TEMP_CONST: {
      return { true, ConstantInt::get(VariableType, Argument->temp->val) };
    }
    default:
      revng_unreachable("unhandled libtcg temp kind");
    }
    break;
  default:
    revng_unreachable("unhandled libtcg arg kind");
  }
}

Value *VariableManager::computeEnvAddress(Type *TargetType,
                                          Instruction *InsertBefore,
                                          unsigned Offset) {
  auto *PointeeTy = Env->getValueType();
  auto *LoadEnv = new LoadInst(PointeeTy, Env, "", InsertBefore);
  Type *EnvType = Env->getValueType();
  Value *Integer = LoadEnv;
  if (Offset != 0)
    Integer = BinaryOperator::Create(Instruction::Add,
                                     LoadEnv,
                                     ConstantInt::get(EnvType, Offset),
                                     "",
                                     InsertBefore);
  return new IntToPtrInst(Integer, TargetType, "", InsertBefore);
}

Value *VariableManager::cpuStateToEnv(Value *CPUState,
                                      Instruction *InsertBefore) const {
  using CI = ConstantInt;

  revng::NonDebugInfoCheckingIRBuilder Builder(InsertBefore);
  auto *OpaquePointer = PointerType::get(TheModule.getContext(), 0);
  auto *IntPtrTy = Builder.getIntPtrTy(TheModule.getDataLayout());
  Value *CPUIntPtr = Builder.CreatePtrToInt(CPUState, IntPtrTy);
  Value *EnvIntPtr = Builder.CreateAdd(CPUIntPtr,
                                       CI::get(IntPtrTy, LibTcgEnvOffset));
  return Builder.CreateIntToPtr(EnvIntPtr, OpaquePointer);
}
