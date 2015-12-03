/// \file
/// \brief This file handles the creation and management of global variables,
///        i.e. mainly parts of the CPU state

// Standard includes
#include <cstdint>
#include <sstream>
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
#include "ptcdump.h"

using namespace llvm;

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
  else
    llvm_unreachable("Unexpected data type");
}

VariableManager::VariableManager(Module& TheModule,
                                 StructType *CPUStateType,
                                 const DataLayout *HelpersModuleLayout) :
  TheModule(TheModule),
  Builder(TheModule.getContext()),
  CPUStateType(CPUStateType),
  HelpersModuleLayout(HelpersModuleLayout),
  Env(nullptr) { }

void VariableManager::newFunction(Instruction *Delimiter,
                                  PTCInstructionList *Instructions) {
  LocalTemporaries.clear();
  newBasicBlock(Delimiter, Instructions);
}

/// Informs the VariableManager that a new basic block has begun, so it can
/// discard basic block-level variables.
///
/// \param Delimiter the new point where to insert allocations for local
/// variables.
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

GlobalVariable* VariableManager::getByCPUStateOffset(intptr_t Offset,
                                                     std::string Name) {

  GlobalsMap::iterator it = CPUStateGlobals.find(Offset);
  if (it != CPUStateGlobals.end()) {
    // TODO: handle renaming
    return it->second;
  } else {
    Type *VariableType = getTypeAtOffset(HelpersModuleLayout,
                                         CPUStateType,
                                         Offset);

    if (Name.size() == 0) {
      std::stringstream NameStream;
      NameStream << "state_0x" << std::hex << Offset;
      Name = NameStream.str();
    }

    auto *NewVariable = new GlobalVariable(TheModule,
                                           VariableType,
                                           false,
                                           GlobalValue::ExternalLinkage,
                                           ConstantInt::get(VariableType, 0),
                                           Name);
    assert(NewVariable != nullptr);
    CPUStateGlobals[Offset] = NewVariable;

    return NewVariable;
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
      return getByCPUStateOffset(Temporary->mem_offset,
                                 StringRef(Temporary->name));
    } else {
      GlobalsMap::iterator it = OtherGlobals.find(TemporaryId);
      if (it != OtherGlobals.end()) {
        return it->second;
      } else {
        auto InitialValue = ConstantInt::get(VariableType, 0);
        GlobalVariable *Result = new GlobalVariable(TheModule,
                                                    VariableType,
                                                    false,
                                                    GlobalValue::ExternalLinkage,
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
