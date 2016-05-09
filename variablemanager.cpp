/// \file
/// \brief This file handles the creation and management of global variables,
///        i.e. mainly parts of the CPU state

// Standard includes
#include <cstdint>
#include <stack>
#include <sstream>
#include <set>
#include <string>

// LLVM includes
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/Casting.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/ValueMapper.h"

// Local includes
#include "debug.h"
#include "ir-helpers.h"
#include "variablemanager.h"
#include "revamb.h"
#include "ptcdump.h"
#include "ptcinterface.h"

using namespace llvm;

#ifndef NDEBUG
namespace llvm {
void Value::assertModuleIsMaterialized() const { }
}
#endif

class OffsetValueStack {

private:
  using OffsetValuePair = std::pair<int64_t, Value *>;

public:
  void pushIfNew(int64_t Offset, Value *V) {
    OffsetValuePair Element = { Offset, V };
    if (!Seen.count(Element)) {
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

static const int64_t ErrorOffset = std::numeric_limits<int64_t>::max();

bool CorrectCPUStateUsagePass::runOnModule(Module& TheModule) {
  OffsetValueStack WorkList;

  Value *CPUStatePtr = TheModule.getGlobalVariable("env");

  // Do we even have "env"?
  if (CPUStatePtr == nullptr)
    return false;

  assert(CPUStatePtr->getType()->isPointerTy());

  struct Specialization {
    Function *F;
    Function *Original;
    std::vector<std::pair<unsigned, int64_t>> SpecializedArgs;
  };

  std::vector<Specialization> Specializations;
  std::map<Function *, int64_t> OffsetFunctions;

  const DataLayout& DL = TheModule.getDataLayout();

  while (true) {
    if (WorkList.empty()) {
      for (Use& CPUStateUse : CPUStatePtr->uses()) {
        auto *Load = cast<LoadInst>(CPUStateUse.getUser());
        assert(Load->getPointerOperand() == CPUStatePtr);

        WorkList.pushIfNew(Variables->EnvOffset, Load);
      }
    }

    if (WorkList.empty())
      break;


    int64_t CurrentOffset;
    Value *CurrentValue;
    std::tie(CurrentOffset, CurrentValue) = WorkList.pop();

    std::vector<std::tuple<User *, Value *, Value *>> Replacements;

    for (Use& TheUse : CurrentValue->uses()) {
      Instruction *TheUser = cast<Instruction>(TheUse.getUser());
      auto Opcode = TheUser->getOpcode();

      if (CurrentOffset == ErrorOffset
          && Opcode != Instruction::Load
          && Opcode != Instruction::Store) {
        // Not loading or storing, propagate the error value
        WorkList.push(ErrorOffset, TheUser);
        continue;
      }

      switch(Opcode) {
      case Instruction::Load:
      case Instruction::Store:
        {
          auto *Load = dyn_cast<LoadInst>(TheUser);
          auto *Store = dyn_cast<StoreInst>(TheUser);

          IRBuilder<> Builder(cast<Instruction>(TheUser));

          bool Success = false;
          if (Load != nullptr) {
            unsigned Size = DL.getTypeSizeInBits(TheUser->getType()) / 8;
            assert(Size != 0);

            unsigned CurrentEnvOffset = CurrentOffset - EnvOffset;
            auto *Loaded = Variables->loadFromEnvOffset(Builder,
                                                        Size,
                                                        CurrentEnvOffset);
            Success = Loaded != nullptr;
            if (Success)
              TheUser->replaceAllUsesWith(Loaded);
          } else {
            Value *ToStore = Store->getValueOperand();
            unsigned Size = DL.getTypeSizeInBits(ToStore->getType()) / 8;
            assert(Size != 0);

            unsigned CurrentEnvOffset = CurrentOffset - EnvOffset;
            Success = Variables->storeToEnvOffset(Builder,
                                                  Size,
                                                  CurrentEnvOffset,
                                                  ToStore);
          }

          if (Success)
            Replacements.push_back(std::make_tuple(TheUser, nullptr, nullptr));
          else
            Builder.CreateCall(TheModule.getFunction("abort"));

          break;
        }
      case Instruction::IntToPtr:
      case Instruction::BitCast:
        {
          // A bitcast, just propagate it
          WorkList.push(CurrentOffset, TheUser);
          break;
        }
      case Instruction::GetElementPtr:
        {
          // A GEP requires to update the offset
          auto *GEP = cast<GetElementPtrInst>(TheUser);
          unsigned AS = GEP->getPointerAddressSpace();
          APInt APOffset(DL.getPointerSizeInBits(AS), 0, true);
          bool Result = GEP->accumulateConstantOffset(DL, APOffset);

          // TODO: do some kind of warning reporting here
          // TODO: split the basic block and add an unreachable here
          if (!Result) {
            CallInst::Create(TheModule.getFunction("abort"), { }, GEP);
            continue;
          }

          int64_t NewOffset = APOffset.getSExtValue();
          WorkList.push(CurrentOffset + NewOffset, TheUser);
          break;
        }
      case Instruction::Add:
        {
          unsigned OtherOperandIndex = 1 - TheUse.getOperandNo();
          Value *OtherOperand = TheUser->getOperand(OtherOperandIndex);

          if (!isa<ConstantInt>(OtherOperand)) {
            auto *InvalidInst = cast<Instruction>(TheUser);
            CallInst::Create(TheModule.getFunction("abort"), { }, InvalidInst);
            continue;
          }

          int64_t Addend = cast<ConstantInt>(OtherOperand)->getSExtValue();
          WorkList.push(CurrentOffset + Addend, TheUser);
          break;
        }
      case Instruction::Call:
        {
          auto *Call = cast<CallInst>(TheUser);
          Function *Callee = Call->getCalledFunction();

          // Some casting with constant expressions?
          if (Callee == nullptr) {
            if (auto *Cast = dyn_cast<ConstantExpr>(Call->getCalledValue())) {
              assert(Cast->getOpcode() == Instruction::BitCast);
              Callee = cast<Function>(Cast->getOperand(0));
            }
          }

          if (Callee != nullptr
              && Callee->getIntrinsicID() == Intrinsic::dbg_declare)
            continue;

          // We only support memcpys where the last parameter is constant
          if (Callee == nullptr
              || (Callee->getIntrinsicID() == Intrinsic::memcpy
                  && !isa<ConstantInt>(Call->getArgOperand(2)))) {
            auto *InvalidInst = cast<Instruction>(TheUser);
            CallInst::Create(TheModule.getFunction("abort"), { }, InvalidInst);
            continue;
          }

          // We're memcpy'ing to the env
          if (Callee->getIntrinsicID() == Intrinsic::memcpy) {
            IRBuilder<> Builder(TheModule.getContext());
            Builder.SetInsertPoint(Call);

            unsigned EnvOpIndex = (Call->getArgOperand(0) == CurrentValue ?
                                   0 : 1);
            Value *BaseOp = Call->getArgOperand(1 - EnvOpIndex);
            auto *ValueOp = cast<Constant>(Call->getArgOperand(2));
            Value *BasePtr = Builder.CreatePtrToInt(BaseOp,
                                                    Builder.getInt64Ty());

            uint64_t TotalSize = getZExtValue(ValueOp, DL);
            uint64_t Offset = 0;

            while (Offset < TotalSize) {
              GlobalVariable *Var = nullptr;
              Var = Variables->getByCPUStateOffset(CurrentOffset + Offset);

              // Consider the case when there's simply nothing there (alignment
              // space)
              if (Var == nullptr) {
                Offset++;
                continue;
              }

              Type *PointeeTy = Var->getType()->getPointerElementType();
              uint64_t Size = DL.getTypeSizeInBits(PointeeTy) / 8;

              Value *Address = Builder.CreateAdd(Builder.getInt64(Offset),
                                                 BasePtr);
              Value *Ptr = Builder.CreateIntToPtr(Address, Var->getType());

              if (EnvOpIndex == 0)
                Builder.CreateStore(Builder.CreateLoad(Ptr), Var);
              else
                Builder.CreateStore(Builder.CreateLoad(Var), Ptr);

              Offset += Size;
            }

            if (Offset != TotalSize) {
              auto *InvalidInstruction = cast<Instruction>(TheUser);
              CallInst::Create(TheModule.getFunction("abort"),
                               { },
                               InvalidInstruction);
              continue;
            }

            // Set memcpy size to 0
            auto *Zero = ConstantInt::get(Call->getArgOperand(2)->getType(), 0);
            Call->setArgOperand(2, Zero);
            continue;
          }

          assert(!Callee->empty() && "external functions are not supported");

          // TODO: move all the specialization-handling code outside
          // Is the callee already a specialization?
          auto Comparison = [&Callee] (Specialization &S) {
            return S.F == Callee;
          };
          auto CurrentSpecialization = std::find_if(Specializations.begin(),
                                                    Specializations.end(),
                                                    Comparison);

          Function *Original = Callee;
          std::vector<std::pair<unsigned, int64_t>> SpecializedArgs;

          // If the callee was already a specialization, preserve its
          // specialized arguments
          if (CurrentSpecialization != Specializations.end()) {
            Original = CurrentSpecialization->Original;
            SpecializedArgs = CurrentSpecialization->SpecializedArgs;
          }

          // Add the new argument to specialize
          SpecializedArgs.push_back({ TheUse.getOperandNo(), CurrentOffset });

          // Does the specialization we want already exists?
          Specialization *Matching = nullptr;
          for (Specialization &S : Specializations) {
            if (S.Original == Original
                && S.SpecializedArgs.size() == SpecializedArgs.size()) {

              Matching = &S;
              for (std::pair<unsigned, int64_t> A : SpecializedArgs) {

                bool Found = false;
                for (std::pair<unsigned, int64_t> B : S.SpecializedArgs) {
                  if (A.first == B.first && A.second == B.second) {
                    Found = true;
                    break;
                  }
                }
                if (!Found) {
                  Matching = nullptr;
                  break;
                }
              }

              if (Matching != nullptr)
                break;

            }
          }

          if (Matching == nullptr) {
            // We need a new specialization
            ValueToValueMapTy VTV;
            SmallVector<ReturnInst *, 5> Returns;

            // Clone existing function
            std::stringstream NewName;
            NewName << Callee->getName().str() << "_" << Specializations.size();
            Callee->setLinkage(GlobalValue::InternalLinkage);
            Function *NewFunc = Function::Create(Callee->getFunctionType(),
                                                 GlobalValue::InternalLinkage,
                                                 NewName.str(),
                                                 Callee->getParent());

            unsigned I = 0;
            auto CalleeArg = Callee->arg_begin();
            auto NewArg = NewFunc->arg_begin();
            for (CalleeArg = Callee->arg_begin();
                 CalleeArg != Callee->arg_end();
                 CalleeArg++) {
              NewArg->setName(CalleeArg->getName());

              WorkList.cloneSisters(&*CalleeArg, &*NewArg);

              VTV[&*CalleeArg] = &*NewArg++;
            }

            CloneFunctionInto(NewFunc, Callee, VTV, true, Returns);

            Specialization New;
            New.F = NewFunc;
            New.Original = Original;
            New.SpecializedArgs = SpecializedArgs;
            Specializations.push_back(New);
            Matching = &Specializations.back();

            // The function is new, we have to explore its argument usage

            // Find the corresponding argument
            auto ArgsI = NewFunc->arg_begin();

            for (I = 0;
                 I < Call->getNumArgOperands() && ArgsI != NewFunc->arg_end();
                 I++, ArgsI++) {
              Use& ArgUse = Call->getArgOperandUse(I);
              if (ArgUse.getOperandNo() == TheUse.getOperandNo())
                break;
            }
            assert(I < Call->getNumArgOperands()
                   && ArgsI != NewFunc->arg_end());

            Value *TargetArg = static_cast<Value *>(&*ArgsI);

            if (TargetArg->use_begin() != TargetArg->use_end()) {
              assert(!NewFunc->isVarArg());

              // If not already considered, enqueue the argument to the worklist
              WorkList.pushIfNew(CurrentOffset, TargetArg);
            }
          }

          auto It = OffsetFunctions.find(Matching->F);
          if (It != OffsetFunctions.end())
            WorkList.push(It->second, static_cast<Value *>(Call));

          auto *OriginalCalleeTy = Call->getCalledValue()->getType();
          Call->setCalledFunction(ConstantExpr::getBitCast(Matching->F,
                                                           OriginalCalleeTy));

          break;
        }
      case Instruction::Ret:
        {
          // This function returns a pointer to the state
          Function *CurrentFunction = TheUser->getParent()->getParent();
          OffsetFunctions[CurrentFunction] = CurrentOffset;
          for (User *FunctionUse : CurrentFunction->users()) {
            auto Call = cast<CallInst>(FunctionUse);
            assert(Call->getCalledFunction() == CurrentFunction);
            WorkList.pushIfNew(CurrentOffset, static_cast<Value *>(Call));
          }
          break;
        }
      default:
        // Unhandled situation, propagate an error value until the next load
        WorkList.push(ErrorOffset, TheUser);
      }
    }

    for (auto Replacement : Replacements)
      if (std::get<1>(Replacement) == nullptr)
        cast<Instruction>(std::get<0>(Replacement))->eraseFromParent();
      else
        std::get<0>(Replacement)->replaceUsesOfWith(std::get<1>(Replacement),
                                                    std::get<2>(Replacement));
  }

  return true;
}

char CorrectCPUStateUsagePass::ID = 0;

static RegisterPass<CorrectCPUStateUsagePass> X("correct-cpustate-usage",
                                                "Correct CPUState Usage Pass",
                                                false,
                                                false);

static std::pair<Type *, unsigned> getTypeAtOffset(const DataLayout *TheLayout,
                                                   StructType *TheStruct,
                                                   intptr_t Offset,
                                                   unsigned Depth=0) {
  const StructLayout *Layout = TheLayout->getStructLayout(TheStruct);
  unsigned FieldIndex = Layout->getElementContainingOffset(Offset);
  uint64_t FieldOffset = Layout->getElementOffset(FieldIndex);
  Type *VariableType = TheStruct->getTypeAtIndex(FieldIndex);
  intptr_t FieldEnd = (FieldOffset
                       + TheLayout->getTypeSizeInBits(VariableType) / 8);

  DBG("type-at-offset", dbg
      << std::string(Depth * 2, ' ')
      << "Offset: " << Offset << " "
      << "Name: " << TheStruct->getName().str() << " "
      << "Index: " << FieldIndex << " "
      << "Field offset: " << FieldOffset << " "
      << "\n");

  if (Offset >= FieldEnd)
    return { nullptr, 0 };

  if (VariableType->isIntegerTy())
    return { VariableType, Offset - FieldOffset };
  else if (VariableType->isArrayTy()) {
    Type *ElementType = VariableType->getArrayElementType();
    uint64_t ElementSize = TheLayout->getTypeSizeInBits(ElementType) / 8;
    if (ElementType->isIntegerTy())
      return { ElementType, (Offset - FieldOffset) % ElementSize };

    return getTypeAtOffset(TheLayout,
                           cast<StructType>(ElementType),
                           (Offset - FieldOffset) % ElementSize,
                           Depth + 1);

  } else if (VariableType->isStructTy())
    return getTypeAtOffset(TheLayout,
                           cast<StructType>(VariableType),
                           Offset - FieldOffset,
                           Depth + 1);
  else {
    // TODO: do some kind of warning reporting here
    return { nullptr, 0 };
  }
}

VariableManager::VariableManager(Module& TheModule,
                                 Module& HelpersModule,
                                 Architecture& TargetArchitecture) :
  TheModule(TheModule),
  Builder(TheModule.getContext()),
  CPUStateType(nullptr),
  ModuleLayout(&HelpersModule.getDataLayout()),
  EnvOffset(0),
  Env(nullptr),
  AliasScopeMDKindID(TheModule.getMDKindID("alias.scope")),
  NoAliasMDKindID(TheModule.getMDKindID("noalias")),
  TargetArchitecture(TargetArchitecture) {

  auto *CPUStateAliasDomain = MDNode::getDistinct(TheModule.getContext(),
                                                  ArrayRef<Metadata *>());

  auto *Temporary = MDNode::get(TheModule.getContext(), ArrayRef<Metadata *>());
  auto *CPUStateScope = MDNode::getDistinct(TheModule.getContext(),
                                            ArrayRef<Metadata *>({
                                                Temporary,
                                                CPUStateAliasDomain
                                            }));
  CPUStateScope->replaceOperandWith(0, CPUStateScope);

  CPUStateScopeSet = MDNode::get(TheModule.getContext(),
                                 ArrayRef<Metadata *>({ CPUStateScope }));

  assert(ptc.initialized_env != nullptr);

  using ElectionMap = std::map<StructType *, unsigned>;
  using ElectionMapElement = std::pair<StructType * const, unsigned>;
  ElectionMap EnvElection;
  const std::string HelperPrefix = "helper_";
  std::set<StructType *> Structs;
  for (Function& HelperFunction : HelpersModule) {
    FunctionType *HelperType = HelperFunction.getFunctionType();
    Type *ReturnType = HelperType->getReturnType();
    if (ReturnType->isPointerTy())
      Structs.insert(dyn_cast<StructType>(ReturnType->getPointerElementType()));

    for (Type *Param : HelperType->params())
      if (Param->isPointerTy())
        Structs.insert(dyn_cast<StructType>(Param->getPointerElementType()));

    if (startsWith(HelperFunction.getName(), HelperPrefix)
        && HelperFunction.getFunctionType()->getNumParams() > 1) {


      for (Type *Candidate : HelperType->params()) {
        Structs.insert(dyn_cast<StructType>(Candidate));
        if (Candidate->isPointerTy()) {
          auto *PointeeType = Candidate->getPointerElementType();
          auto *EnvType = dyn_cast<StructType>(PointeeType);
          // Ensure it is a struct and not a union
          if (EnvType != nullptr && EnvType->getNumElements() > 1) {

            auto It = EnvElection.find(EnvType);
            if (It != EnvElection.end())
              EnvElection[EnvType]++;
            else
              EnvElection[EnvType] = 1;
          }
        }
      }
    }
  }

  Structs.erase(nullptr);

  assert(EnvElection.size() > 0);

  CPUStateType = std::max_element(EnvElection.begin(),
                                  EnvElection.end(),
                                  [] (ElectionMapElement& It1,
                                      ElectionMapElement& It2) {
                                    return It1.second < It2.second;
                                  })->first;

  // Look for structures containing CPUStateType as a member and promove them
  // to CPUStateType. Basically this is a flexible way to keep track of the *CPU
  // struct too (e.g. MIPSCPU).
  std::set<StructType *> Visited;
  bool Changed = true;
  Visited.insert(CPUStateType);
  while (Changed) {
    Changed = false;
    for (StructType *TheStruct : Structs) {
      if (Visited.find(TheStruct) != Visited.end())
        continue;

      auto Begin = TheStruct->element_begin();
      auto End = TheStruct->element_end();
      auto Found = std::find(Begin, End, CPUStateType);
      if (Found != End) {
        unsigned Index = Found - Begin;
        const StructLayout *Layout = nullptr;
        Layout = ModuleLayout->getStructLayout(TheStruct);
        EnvOffset += Layout->getElementOffset(Index);
        CPUStateType = TheStruct;
        Visited.insert(CPUStateType);
        Changed = true;
        break;
      }
    }
  }
}

bool VariableManager::storeToCPUStateOffset(IRBuilder<> &Builder,
                                            unsigned StoreSize,
                                            unsigned Offset,
                                            Value *ToStore) {
  Value *Target;
  unsigned Remaining;
  std::tie(Target, Remaining) = getByCPUStateOffsetInternal(Offset);

  assert(Target != nullptr);

  if (Target == nullptr)
    return false;

  unsigned ShiftAmount = 0;
  if (TargetArchitecture.isLittleEndian())
    ShiftAmount = Remaining;
  else {
    // >> (Size1 - Size2) - Remaining;
    Type *PointeeTy = Target->getType()->getPointerElementType();
    unsigned GlobalSize = cast<IntegerType>(PointeeTy)->getBitWidth() / 8;
    assert(GlobalSize != 0);
    ShiftAmount = (GlobalSize - StoreSize) - Remaining;
  }
  ShiftAmount *= 8;

  // Build blanking mask
  uint64_t BitMask = (StoreSize == 8 ?
                      (uint64_t) -1
                      : ((uint64_t) 1 << StoreSize * 8) - 1);
  assert(ShiftAmount != 64);
  BitMask <<= ShiftAmount;
  BitMask = ~BitMask;

  auto *InputStoreTy = cast<IntegerType>(Builder.getIntNTy(StoreSize * 8));
  auto *FieldTy = cast<IntegerType>(Target->getType()->getPointerElementType());
  unsigned FieldSize = FieldTy->getBitWidth() / 8;

  // Truncate value to store
  auto *Truncated = Builder.CreateTrunc(ToStore, InputStoreTy);

  // Are we trying to store more than it fits?
  if (StoreSize > FieldSize) {
    // It's OK as long as after what we're storing there's a hole
    assert(getByCPUStateOffsetInternal(Offset + FieldSize).first == nullptr);
    Truncated = Builder.CreateTrunc(Truncated, FieldTy);
  }

  // Re-extend
  ToStore = Builder.CreateZExt(Truncated, FieldTy);

  if (BitMask != 0) {
    // Load the value
    auto *LoadEnvField = Builder.CreateLoad(Target);
    setAliasScope(LoadEnvField);

    auto *Blanked = Builder.CreateAnd(LoadEnvField, BitMask);

    // Shift value to store
    ToStore = Builder.CreateShl(ToStore, ShiftAmount);

    // Combine them
    ToStore = Builder.CreateOr(ToStore, Blanked);
  }

  // Type *TargetPointer = Target->getType()->getPointerElementType();
  // Value *ToStore = Builder.CreateZExt(InArguments[0], TargetPointer);
  auto *Store = Builder.CreateStore(ToStore, Target);
  setAliasScope(Store);

  return true;
}

Value *VariableManager::loadFromCPUStateOffset(IRBuilder<> &Builder,
                                               unsigned LoadSize,
                                               unsigned Offset) {
  Value *Target;
  unsigned Remaining;
  std::tie(Target, Remaining) = getByCPUStateOffsetInternal(Offset);

  if (Target == nullptr)
    return nullptr;

  // Load the whole field
  auto *LoadEnvField = Builder.CreateLoad(Target);
  setAliasScope(LoadEnvField);

  // Extract the desired part
  // Shift right of the desired amount
  unsigned ShiftAmount = 0;
  if (TargetArchitecture.isLittleEndian())
    ShiftAmount = Remaining;
  else {
    // >> (Size1 - Size2) - Remaining;
    auto *LoadedTy = cast<IntegerType>(LoadEnvField->getType());
    unsigned GlobalSize = LoadedTy->getBitWidth() / 8;
    assert(GlobalSize != 0);
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
      // It's OK as long as after what we can't load there's a hole
      assert(getByCPUStateOffsetInternal(Offset + FieldSize).first == nullptr);
      Result = Builder.CreateZExt(Result, LoadTy);
    }
  }

  // Truncate of the desired amount
  return Builder.CreateTrunc(Result, LoadTy);
}

// TODO: `newFunction` reflects the tcg terminology but in this context is
//       highly misleading
void VariableManager::newFunction(Instruction *Delimiter,
                                  PTCInstructionList *Instructions) {
  LocalTemporaries.clear();
  newBasicBlock(Delimiter, Instructions);
}

/// Informs the VariableManager that a new basic block has begun, so it can
/// discard basic block-level variables.
///
/// \param Delimiter the new point where to insert allocations for local
///                  variables.
/// \param Instructions the new PTCInstructionList to use from now on.
void VariableManager::newBasicBlock(Instruction *Delimiter,
                                    PTCInstructionList *Instructions) {
  Temporaries.clear();
  if (Instructions != nullptr)
    this->Instructions = Instructions;

  if (Delimiter != nullptr)
    Builder.SetInsertPoint(Delimiter);
}

void VariableManager::newBasicBlock(BasicBlock *Delimiter,
                                    PTCInstructionList *Instructions) {
  Temporaries.clear();
  if (Instructions != nullptr)
    this->Instructions = Instructions;

  if (Delimiter != nullptr)
    Builder.SetInsertPoint(Delimiter);
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

  llvm_unreachable("Unexpected type");
}

// TODO: document that it can return nullptr
GlobalVariable* VariableManager::getByCPUStateOffset(intptr_t Offset,
                                                     std::string Name) {
  GlobalVariable *Result = nullptr;
  unsigned Remaining;
  std::tie(Result, Remaining) = getByCPUStateOffsetInternal(Offset, Name);
  assert(Remaining == 0);
  return Result;
}

std::pair<GlobalVariable*, unsigned>
VariableManager::getByCPUStateOffsetInternal(intptr_t Offset,
                                             std::string Name) {
  if (Offset == ErrorOffset)
    return { nullptr, 0 };

  GlobalsMap::iterator it = CPUStateGlobals.find(Offset);
  if (it == CPUStateGlobals.end() ||
      (Name.size() != 0 && !it->second->getName().equals_lower(Name))) {
    Type *VariableType;
    unsigned Remaining;
    std::tie(VariableType, Remaining) = getTypeAtOffset(ModuleLayout,
                                                        CPUStateType,
                                                        Offset);

    // Check we're not trying to go inside an existing variable
    if (Remaining != 0) {
      GlobalsMap::iterator it = CPUStateGlobals.find(Offset - Remaining);
      if (it != CPUStateGlobals.end())
        return { it->second, Remaining };
    }

    // Unsupported type, let the caller handle the situation
    if (VariableType == nullptr)
      return { nullptr, 0 };

    if (Name.size() == 0) {
      std::stringstream NameStream;
      NameStream << "state_0x" << std::hex << Offset;
      Name = NameStream.str();
    }

    // TODO: offset could be negative, we could segfault here
    auto *InitialValue = fromBytes(cast<IntegerType>(VariableType),
                                   ptc.initialized_env - EnvOffset + Offset);

    auto *NewVariable = new GlobalVariable(TheModule,
                                           VariableType,
                                           false,
                                           GlobalValue::ExternalLinkage,
                                           InitialValue,
                                           Name);
    assert(NewVariable != nullptr);

    if (it != CPUStateGlobals.end()) {
      it->second->replaceAllUsesWith(NewVariable);
      it->second->eraseFromParent();
    }

    CPUStateGlobals[Offset] = NewVariable;

    return { NewVariable, Remaining };
  } else {
    return { it->second, 0 };
  }
}

Value *VariableManager::getOrCreate(unsigned TemporaryId, bool Reading) {
  assert(Instructions != nullptr);

  PTCTemp *Temporary = ptc_temp_get(Instructions, TemporaryId);
  Type *VariableType = Temporary->type == PTC_TYPE_I32 ?
    Builder.getInt32Ty() : Builder.getInt64Ty();

  if (ptc_temp_is_global(Instructions, TemporaryId)) {
    // Basically we use fixed_reg to detect "env"
    if (Temporary->fixed_reg == 0) {
      Value *Result = getByCPUStateOffset(EnvOffset + Temporary->mem_offset,
                                          StringRef(Temporary->name));
      assert(Result != nullptr);
      return Result;
    } else {
      GlobalsMap::iterator it = OtherGlobals.find(TemporaryId);
      if (it != OtherGlobals.end()) {
        return it->second;
      } else {
        // TODO: what do we have here, apart from env?
        auto InitialValue = ConstantInt::get(VariableType, 0);
        GlobalVariable *Result = new GlobalVariable(TheModule,
                                                    VariableType,
                                                    false,
                                                    GlobalValue::CommonLinkage,
                                                    InitialValue,
                                                    StringRef(Temporary->name));

        if (Result->getName() == "env")
          Env = Result;

        OtherGlobals[TemporaryId] = Result;
        return Result;
      }
    }
  } else if (Temporary->temp_local) {
    auto it = LocalTemporaries.find(TemporaryId);
    if (it != LocalTemporaries.end()) {
      return it->second;
    } else {
      AllocaInst *NewTemporary = Builder.CreateAlloca(VariableType);
      LocalTemporaries[TemporaryId] = NewTemporary;
      return NewTemporary;
    }
  } else {
    auto it = Temporaries.find(TemporaryId);
    if (it != Temporaries.end()) {
      return it->second;
    } else {
      // Can't read a temporary if it has never been written, we're probably
      // translating rubbish
      if (Reading)
        return nullptr;

      AllocaInst *NewTemporary = Builder.CreateAlloca(VariableType);
      Temporaries[TemporaryId] = NewTemporary;
      return NewTemporary;
    }
  }
}

template LoadInst *VariableManager::setAliasScope(LoadInst *);
template StoreInst *VariableManager::setAliasScope(StoreInst *);

template<typename T>
T *VariableManager::setAliasScope(T *Instruction) {
  auto *Pointer = Instruction->getPointerOperand();
  if (isa<AllocaInst>(Pointer))
    return Instruction;

  Instruction->setMetadata(AliasScopeMDKindID, CPUStateScopeSet);
  return Instruction;
}

template LoadInst *VariableManager::setNoAlias(LoadInst *);
template StoreInst *VariableManager::setNoAlias(StoreInst *);

template<typename T>
T *VariableManager::setNoAlias(T *Instruction) {
  Instruction->setMetadata(NoAliasMDKindID, CPUStateScopeSet);
  return Instruction;
}

Value *VariableManager::computeEnvAddress(Type *TargetType,
                                          Instruction *InsertBefore,
                                          unsigned Offset) {
  auto *LoadEnv = new LoadInst(Env, "", InsertBefore);
  Type *EnvType = Env->getType()->getPointerElementType();
  Value *Integer = LoadEnv;
  if (Offset != 0)
    Integer =  BinaryOperator::Create(Instruction::Add,
                                      LoadEnv,
                                      ConstantInt::get(EnvType, Offset),
                                      "",
                                      InsertBefore);
  return new IntToPtrInst(Integer, TargetType, "", InsertBefore);
}
