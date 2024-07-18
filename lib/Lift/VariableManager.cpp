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

#include "revng/Model/Register.h"
#include "revng/Support/Assert.h"
#include "revng/Support/Debug.h"
#include "revng/Support/FunctionTags.h"
#include "revng/Support/IRHelpers.h"

#include "VariableManager.h"

RegisterIRHelper SetRegisterMarker("set_register", "in early-linked");

using namespace llvm;

// TODO: rename
cl::opt<bool> External("external",
                       cl::desc("set CSVs linkage to external, useful for "
                                "debugging purposes"),
                       cl::cat(MainCategory));
static cl::alias A1("E",
                    cl::desc("Alias for -external"),
                    cl::aliasopt(External),
                    cl::cat(MainCategory));

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

static Logger<> Log("type-at-offset");

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
      return { cast<IntegerType>(VarType), Offset };

    case llvm::Type::TypeID::ArrayTyID: {
      VarType = VarType->getArrayElementType();
      auto TypeSize = TheLayout->getTypeAllocSize(VarType);
      auto Index = Offset / TypeSize;
      Offset %= TypeSize;
      revng_log(Log,
                Prefix << "Element " << Index
                       << " in an array. Offset in Element: " << Offset);
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
                                 StructType *ArchCPUStruct,
                                 unsigned LibTcgEnvOffset,
                                 uint8_t *LibTcgEnvPtr) :
  TheModule(M),
  AllocaBuilder(getContext(&M)),
  ArchCPUStruct(ArchCPUStruct),
  ModuleLayout(&TheModule.getDataLayout()),
  LibTcgEnvOffset(LibTcgEnvOffset),
  LibTcgEnvPtr(LibTcgEnvPtr),
  Env(nullptr),
  TargetIsLittleEndian(TargetIsLittleEndian) {

  revng_assert(LibTcgEnvPtr != nullptr);

  IntegerType *IntPtrTy = AllocaBuilder.getIntPtrTy(*ModuleLayout);
  Env = cast<GlobalVariable>(TheModule.getOrInsertGlobal("env", IntPtrTy));
  Env->setInitializer(ConstantInt::get(IntPtrTy, LibTcgEnvOffset));
}

std::optional<StoreInst *>
VariableManager::storeToCPUStateOffset(IRBuilder<> &Builder,
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

Value *VariableManager::loadFromCPUStateOffset(IRBuilder<> &Builder,
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

void VariableManager::memOpAtEnvOffset(llvm::IRBuilder<> &Builder,
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

  if (not External) {
    for (auto &P : CPUStateGlobals)
      P.second->setLinkage(GlobalValue::InternalLinkage);
    for (auto &P : OtherGlobals)
      P.second->setLinkage(GlobalValue::InternalLinkage);
  }

  IRBuilder<> Builder(Context);

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
GlobalVariable *VariableManager::getByCPUStateOffset(intptr_t Offset,
                                                     std::string Name) {
  GlobalVariable *Result = nullptr;
  unsigned Remaining;
  std::tie(Result, Remaining) = getByCPUStateOffsetWithRemainder(Offset, Name);
  revng_assert(Remaining == 0);
  return Result;
}

std::pair<GlobalVariable *, unsigned>
VariableManager::getByCPUStateOffsetWithRemainder(intptr_t Offset,
                                                  std::string Name) {
  GlobalsMap::iterator It = CPUStateGlobals.find(Offset);
  static const char *UnknownCSVPref = "state_0x";
  if (It == CPUStateGlobals.end()
      || (Name.size() != 0
          && It->second->getName().startswith(UnknownCSVPref))) {
    Type *VariableType = nullptr;
    unsigned Remaining;
    std::tie(VariableType,
             Remaining) = getTypeAtOffset(ModuleLayout, ArchCPUStruct, Offset);

    // Unsupported type, let the caller handle the situation
    if (VariableType == nullptr)
      return { nullptr, 0 };

    // Check we're not trying to go inside an existing variable
    if (Remaining != 0) {
      GlobalsMap::iterator It = CPUStateGlobals.find(Offset - Remaining);
      if (It != CPUStateGlobals.end())
        return { It->second, Remaining };
    }

    if (Name.size() == 0) {
      std::stringstream NameStream;
      NameStream << UnknownCSVPref << std::hex << Offset;
      Name = NameStream.str();
    }

    // TODO: offset could be negative, we could segfault here
    auto *InitialValue = fromBytes(cast<IntegerType>(VariableType),
                                   LibTcgEnvPtr - LibTcgEnvOffset + Offset);

    auto *NewVariable = new GlobalVariable(TheModule,
                                           VariableType,
                                           false,
                                           GlobalValue::ExternalLinkage,
                                           InitialValue,
                                           Name);

    auto Register = model::Register::fromCSVName(Name,
                                                 model::Architecture::Invalid);
    if (Register != model::Register::Invalid) {
      auto ExpectedSize = model::Register::getSize(Register);
      auto ActualSize = VariableType->getIntegerBitWidth();
      revng_assert(ExpectedSize == ActualSize);
    }

    revng_assert(NewVariable != nullptr);
    FunctionTags::CSV.addTo(NewVariable);

    if (It != CPUStateGlobals.end()) {
      It->second->replaceAllUsesWith(NewVariable);
      eraseFromParent(It->second);
    }

    CPUStateGlobals[Offset] = NewVariable;

    return { NewVariable, Remaining };
  } else {
    return { It->second, 0 };
  }
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
                                            + Argument->temp->mem_offset,
                                          Argument->temp->name);
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

  IRBuilder<> Builder(InsertBefore);
  auto *OpaquePointer = PointerType::get(TheModule.getContext(), 0);
  auto *IntPtrTy = Builder.getIntPtrTy(TheModule.getDataLayout());
  Value *CPUIntPtr = Builder.CreatePtrToInt(CPUState, IntPtrTy);
  Value *EnvIntPtr = Builder.CreateAdd(CPUIntPtr,
                                       CI::get(IntPtrTy, LibTcgEnvOffset));
  return Builder.CreateIntToPtr(EnvIntPtr, OpaquePointer);
}
