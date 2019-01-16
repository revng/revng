#ifndef MEMORYACCESS_H
#define MEMORYACCESS_H

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// Standard includes
#include <cstdint>
#include <unordered_map>
#include <utility>

// LLVM includes
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/Support/Casting.h"

// Local libraries includes
#include "revng/Support/IRHelpers.h"

class TypeSizeProvider {
public:
  TypeSizeProvider(const llvm::DataLayout &DL) : DL(DL) {}

  unsigned getSize(llvm::Type *T) {
    auto CacheIt = Cache.find(T);
    if (CacheIt != Cache.end()) {
      return CacheIt->second;
    } else {
      auto Result = DL.getTypeAllocSize(T);
      Cache[T] = Result;
      return Result;
    }
  }

private:
  std::unordered_map<llvm::Type *, unsigned> Cache;
  const llvm::DataLayout &DL;
};

/// \brief Represents an access to the CPU state or the memory
class MemoryAccess {
public:
  MemoryAccess() : Type(Invalid), Base(nullptr), Offset(0), Size(0) {}

  MemoryAccess(llvm::Instruction *I, TypeSizeProvider &TSP) {
    if (auto *Load = llvm::dyn_cast<llvm::LoadInst>(I)) {
      initialize(Load->getPointerOperand(), I, TSP);
    } else if (auto *Store = llvm::dyn_cast<llvm::StoreInst>(I)) {
      initialize(Store->getPointerOperand(), Store->getValueOperand(), TSP);
    } else {
      revng_abort();
    }
  }

  MemoryAccess(llvm::LoadInst *Load, TypeSizeProvider &TSP) {
    initialize(Load->getPointerOperand(), Load, TSP);
  }

  MemoryAccess(llvm::StoreInst *Store, TypeSizeProvider &TSP) {
    initialize(Store->getPointerOperand(), Store->getValueOperand(), TSP);
  }

  MemoryAccess(llvm::Instruction *I, const llvm::DataLayout &DL) {
    if (auto *Load = llvm::dyn_cast<llvm::LoadInst>(I)) {
      initialize(Load->getPointerOperand(), I, DL);
    } else if (auto *Store = llvm::dyn_cast<llvm::StoreInst>(I)) {
      initialize(Store->getPointerOperand(), Store->getValueOperand(), DL);
    } else {
      revng_abort();
    }
  }

  MemoryAccess(llvm::LoadInst *Load, const llvm::DataLayout &DL) {
    initialize(Load->getPointerOperand(), Load, DL);
  }

  MemoryAccess(llvm::StoreInst *Store, const llvm::DataLayout &DL) {
    initialize(Store->getPointerOperand(), Store->getValueOperand(), DL);
  }

  bool operator==(const MemoryAccess &Other) const {
    if (Type != Other.Type or Size != Other.Size)
      return false;

    switch (Type) {
    case Invalid:
      return true;
    case CPUState:
      return Base == Other.Base;
    case RegisterAndOffset:
      return Base == Other.Base && Offset == Other.Offset;
    case Absolute:
      return Offset == Other.Offset;
    }

    revng_abort();
  }

  bool operator!=(const MemoryAccess &Other) const { return !(*this == Other); }

  bool mayAlias(const MemoryAccess &Other) const {
    // If they are exactly the same, they do alias
    if (*this == Other)
      return true;

    // TODO: is this correct?
    if (Type == Invalid || Other.Type == Invalid)
      return true;

    // If they're both CPU state, they alias only if the are the same part of
    // the CPU state. If one of them is CPU state and the other is a register +
    // offset, they alias only if the register written by the first memory
    // access is the one read by the second one.
    if ((Type == CPUState && Other.Type == CPUState)
        || (Type == CPUState && Other.Type == RegisterAndOffset)
        || (Type == RegisterAndOffset && Other.Type == CPUState))
      return Base == Other.Base;

    // If they're RegisterAndOffset and they're not relative to the same
    // register we known nothing about the content of the base register,
    // therefore they may alias.
    // If they're relative to the same register, we check if the two memory
    // accesses overlap, if they don't there's no alias.
    // Note that we can assume the content of the register is the same, since if
    // this wasn't the case we'd have already had an alias situation when
    // writing the register.
    if (Type == RegisterAndOffset && Other.Type == RegisterAndOffset) {
      if (Base != Other.Base)
        return true;

      return intersect({ Offset, Size }, { Other.Offset, Other.Size });
    }

    // Absolute addresses and CPUState never alias
    if ((Type == Absolute and Other.Type == CPUState)
        or (Type == CPUState and Other.Type == Absolute))
      return false;

    // Absolute addresses and RegisterAndOffset may always alias
    if ((Type == Absolute and Other.Type == RegisterAndOffset)
        or (Type == RegisterAndOffset and Other.Type == Absolute))
      return true;

    // We have two absolute ranges, do they intersect?
    if (Type == Absolute and Other.Type == Absolute)
      return intersect({ Offset, Size }, { Other.Offset, Other.Size });

    revng_abort();
  }

  bool isValid() const { return Type != Invalid; }

  static bool mayAlias(llvm::BasicBlock *BB,
                       const MemoryAccess &Other,
                       const llvm::DataLayout &DL) {
    for (llvm::Instruction &I : *BB)
      if (auto *Store = llvm::dyn_cast<llvm::StoreInst>(&I))
        if (MemoryAccess(Store, DL).mayAlias(Other))
          return true;

    return false;
  }

public:
  llvm::Value *globalVariable() const {
    if (Type == CPUState)
      return Base;
    else
      return nullptr;
  }

  llvm::Value *base() const {
    if (Type == RegisterAndOffset)
      return Base;
    else
      return nullptr;
  }

private:
  bool intersect(std::pair<uint64_t, uint64_t> A,
                 std::pair<uint64_t, uint64_t> B) const {
    return A.first < (B.first + B.second) && B.first < (A.first + A.second);
  }

  bool isVariable(llvm::Value *V) const {
    auto *CSV = llvm::dyn_cast<llvm::GlobalVariable>(V);
    return (CSV != nullptr && CSV->getName() != "env")
           || llvm::isa<llvm::AllocaInst>(V);
  }

  void initialize(llvm::Value *Pointer,
                  llvm::Value *PointeeValue,
                  const llvm::DataLayout &DL) {
    // Set the size
    Size = DL.getTypeAllocSize(PointeeValue->getType());
    initialize(Pointer);
  }

  void initialize(llvm::Value *Pointer,
                  llvm::Value *PointeeValue,
                  TypeSizeProvider &TSP) {
    // Set the size
    Size = TSP.getSize(PointeeValue->getType());
    initialize(Pointer);
  }

  void initialize(llvm::Value *Pointer) {
    // Default situation: we can't handle this load
    Type = Invalid;
    Base = nullptr;
    Offset = 0;

    Pointer = skipCasts(Pointer);

    if (auto *C = llvm::dyn_cast<llvm::ConstantInt>(Pointer)) {
      // Load from an absolute address
      Type = Absolute;
      Offset = C->getLimitedValue();

    } else if (isVariable(Pointer)) {
      // Load from CPU state
      Type = CPUState;
      Base = Pointer;

    } else if (auto *V = llvm::dyn_cast<llvm::Instruction>(Pointer)) {
      // Try to handle load from an address stored in a register plus an offset
      // This mainly aims to handle very simple variables stored on the stack
      uint64_t Addend = 0;
      while (true) {
        switch (V->getOpcode()) {
        case llvm::Instruction::IntToPtr:
        case llvm::Instruction::PtrToInt: {
          auto *Operand = llvm::dyn_cast<llvm::Instruction>(V->getOperand(0));
          if (Operand != nullptr)
            V = Operand;
          else
            return;
        } break;
        case llvm::Instruction::Add: {
          auto Operands = operandsByType<llvm::Instruction *,
                                         llvm::ConstantInt *>(V);
          llvm::Instruction *FirstOp;
          llvm::ConstantInt *SecondOp;
          std::tie(FirstOp, SecondOp) = Operands;
          if (Addend != 0 || SecondOp == nullptr || FirstOp == nullptr)
            return;

          Addend = SecondOp->getLimitedValue();
          V = FirstOp;

        } break;
        case llvm::Instruction::Load: {
          llvm::Value *LoadOperand = V->getOperand(0);
          if (isVariable(LoadOperand)) {
            Type = RegisterAndOffset;
            Base = LoadOperand;
            Offset = Addend;
          }
        }
          return;
        default:
          return;
        }
      }
    }
  }

private:
  enum { Invalid, CPUState, RegisterAndOffset, Absolute } Type;
  llvm::Value *Base;
  uint64_t Offset;
  uint64_t Size;
};

#endif // MEMORYACCESS_H
