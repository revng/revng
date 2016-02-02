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
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/Casting.h"

// Local includes
#include "ir-helpers.h"
#include "variablemanager.h"
#include "revamb.h"
#include "ptcdump.h"
#include "ptcinterface.h"

using namespace llvm;

template<typename T>
static void pushIfNew(std::set<T>& Seen, std::stack<T>& Queue, T Element) {
  if (Seen.find(Element) == Seen.end()) {
    Seen.insert(Element);
    Queue.push(Element);
  }
}

static const int64_t ErrorOffset = std::numeric_limits<int64_t>::max();

bool CorrectCPUStateUsagePass::runOnModule(Module& TheModule) {
  using OffsetValuePair = std::pair<int64_t, Value *>;
  std::set<OffsetValuePair> SeenArgs;
  std::stack<OffsetValuePair> WorkList;

  Value *CPUStatePtr = TheModule.getGlobalVariable("env");

  // Do we even have "env"?
  if (CPUStatePtr == nullptr)
    return false;

  assert(CPUStatePtr->getType()->isPointerTy());

  // Initialize the worklist with all the instructions loading env
  for (Use& CPUStateUse : CPUStatePtr->uses()) {
    auto *Load = cast<LoadInst>(CPUStateUse.getUser());
    assert(Load->getPointerOperand() == CPUStatePtr);

    WorkList.push(std::make_pair(Variables->EnvOffset, Load));
  }

  const DataLayout& DL = TheModule.getDataLayout();

  while (!WorkList.empty()) {
    int64_t CurrentOffset;
    Value *CurrentValue;
    std::tie(CurrentOffset, CurrentValue) = WorkList.top();
    WorkList.pop();

    std::vector<std::tuple<User *, Value *, Value *>> Replacements;

    for (Use& TheUse : CurrentValue->uses()) {
      Instruction *TheUser = cast<Instruction>(TheUse.getUser());
      auto Opcode = TheUser->getOpcode();

      if (CurrentOffset == ErrorOffset
          && Opcode != Instruction::Load
          && Opcode != Instruction::Store) {
        // Not loading or storing, propagate the error value
        WorkList.push(std::make_pair(ErrorOffset, TheUser));
        continue;
      }

      switch(Opcode) {
      case Instruction::Load:
      case Instruction::Store:
        {
          if (Opcode == Instruction::Store) {
            // It's a store, just change the destination pointer
            assert(cast<StoreInst>(TheUser)->getPointerOperand() == CurrentValue
                   && "Pointer cannot be used as source of a store instruction");
          } else if (Opcode == Instruction::Load) {
            // It's a load, just change the source pointer
            assert(cast<LoadInst>(TheUser)->getPointerOperand() == CurrentValue
                   && "Pointer cannot be used as destination of a load"
                   " instruction");
          }

          GlobalVariable *Var = Variables->getByCPUStateOffset(CurrentOffset);

          // Couldn't translate this environment usage, make it fail at run-time
          if (Var == nullptr) {
            auto *InvalidInst = cast<Instruction>(TheUser);
            // TODO: emit a warning
            CallInst::Create(TheModule.getFunction("abort"), { }, InvalidInst);
            // TODO: shall we put an unreachable and delete everything comes
            //       afterwards?
          } else {
            Constant *Ptr = Var;

            // Sadly, we have to allow this, mainly due to unions
            if (CurrentValue->getType() != Var->getType())
              Ptr = ConstantExpr::getPointerCast(Ptr, CurrentValue->getType());

            Replacements.push_back(std::make_tuple(TheUser, CurrentValue, Ptr));
          }
          break;
        }
      case Instruction::IntToPtr:
      case Instruction::BitCast:
        {
          // A bitcast, just propagate it
          WorkList.push(std::make_pair(CurrentOffset, TheUser));
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
          WorkList.push(std::make_pair(CurrentOffset + NewOffset, TheUser));
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
          WorkList.push(std::make_pair(CurrentOffset + Addend, TheUser));
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

          // TODO: we could handle this instead of aborting
          if (Callee == nullptr
              || Callee->getIntrinsicID() == Intrinsic::memcpy) {
            auto *InvalidInst = cast<Instruction>(TheUser);
            CallInst::Create(TheModule.getFunction("abort"), { }, InvalidInst);
            continue;
          }

          assert(!Callee->empty() && "external functions are not supported");

          // Find the corresponding argument
          auto ArgsI = Callee->arg_begin();
          unsigned I = 0;

          for (I = 0;
               I < Call->getNumArgOperands() && ArgsI != Callee->arg_end();
               I++, ArgsI++) {
            Use& ArgUse = Call->getArgOperandUse(I);
            if (ArgUse.getOperandNo() == TheUse.getOperandNo())
              break;
          }
          assert(I < Call->getNumArgOperands()
                 && ArgsI != Callee->arg_end());

          Value *TargetArg = static_cast<Value *>(&*ArgsI);

          if (TargetArg->use_begin() != TargetArg->use_end()) {
            assert(!Callee->isVarArg());

            // If not already considered, enqueue the argument to the worklist
            pushIfNew(SeenArgs,
                      WorkList,
                      std::make_pair(CurrentOffset, TargetArg));
          }
          break;
        }
      case Instruction::Ret:
        {
          // This function returns a pointer to the state
          Function *CurrentFunction = TheUser->getParent()->getParent();
          for (User *FunctionUse : CurrentFunction->users()) {
            auto Call = cast<CallInst>(FunctionUse);
            assert(Call->getCalledFunction() == CurrentFunction);
            pushIfNew(SeenArgs,
                      WorkList,
                      std::make_pair(CurrentOffset,
                                     static_cast<Value *>(Call)));
          }
          break;
        }
      default:
        // Unhandled situation, propagate an error value until the next load
        WorkList.push(std::make_pair(ErrorOffset, TheUser));
      }
    }

    for (auto Replacement : Replacements)
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

static Type *getTypeAtOffset(const DataLayout *TheLayout,
                             StructType *TheStruct,
                             intptr_t Offset) {
  const StructLayout *Layout = TheLayout->getStructLayout(TheStruct);
  unsigned FieldIndex = Layout->getElementContainingOffset(Offset);
  uint64_t FieldOffset = Layout->getElementOffset(FieldIndex);

  Type *VariableType = TheStruct->getTypeAtIndex(FieldIndex);

  if (VariableType->isIntegerTy())
    return VariableType;
  else if (VariableType->isArrayTy()) {
    Type *ElementType = VariableType->getArrayElementType();
    if (ElementType->isIntegerTy())
      return ElementType;

    uint64_t ElementSize = TheLayout->getTypeSizeInBits(ElementType) / 8;
    return getTypeAtOffset(TheLayout,
                           cast<StructType>(ElementType),
                           (Offset - FieldOffset) % ElementSize);

  } else if (VariableType->isStructTy())
    return getTypeAtOffset(TheLayout,
                           cast<StructType>(VariableType),
                           Offset - FieldOffset);
  else {
    // TODO: do some kind of warning reporting here
    return nullptr;
  }
}

VariableManager::VariableManager(Module& TheModule,
                                 Module& HelpersModule) :
  TheModule(TheModule),
  Builder(TheModule.getContext()),
  CPUStateType(nullptr),
  ModuleLayout(&HelpersModule.getDataLayout()),
  EnvOffset(0),
  Env(nullptr),
  AliasScopeMDKindID(TheModule.getMDKindID("alias.scope")),
  NoAliasMDKindID(TheModule.getMDKindID("noalias")) {

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

    for (Type *Candidate : HelperType->params())
      if (Candidate->isPointerTy())
        Structs.insert(dyn_cast<StructType>(Candidate->getPointerElementType()));

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

  assert(false && "Unexpected type");
}

// TODO: document that it can return nullptr
GlobalVariable* VariableManager::getByCPUStateOffset(intptr_t Offset,
                                                     std::string Name) {
  if (Offset == ErrorOffset)
    return nullptr;

  GlobalsMap::iterator it = CPUStateGlobals.find(Offset);
  if (it == CPUStateGlobals.end() ||
      (Name.size() != 0 && !it->second->getName().equals_lower(Name))) {
    Type *VariableType = getTypeAtOffset(ModuleLayout,
                                         CPUStateType,
                                         Offset);

    // Unsupported type, let the caller handle the situation
    if (VariableType == nullptr)
      return nullptr;

    if (Name.size() == 0) {
      std::stringstream NameStream;
      NameStream << "state_0x" << std::hex << Offset;
      Name = NameStream.str();
    }

    // TODO: offset could be negative, we could segfault here
    ConstantInt *InitialValue = fromBytes(cast<IntegerType>(VariableType),
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

    return NewVariable;
  } else {
    return it->second;
  }
}

Value* VariableManager::getOrCreate(unsigned int TemporaryId) {
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
    TemporariesMap::iterator it = LocalTemporaries.find(TemporaryId);
    if (it != LocalTemporaries.end()) {
      return it->second;
    } else {
      AllocaInst *NewTemporary = Builder.CreateAlloca(VariableType);
      LocalTemporaries[TemporaryId] = NewTemporary;
      return NewTemporary;
    }
  } else {
    TemporariesMap::iterator it = Temporaries.find(TemporaryId);
    if (it != Temporaries.end()) {
      return it->second;
    } else {
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
