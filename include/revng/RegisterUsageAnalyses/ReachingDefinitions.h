#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/SmallVector.h"

#include "revng/MFP/MFP.h"
#include "revng/RegisterUsageAnalyses/Function.h"

namespace rua {

struct RegisterWriters {
  /// Set of writes that reach this point alive
  llvm::BitVector Reaching;

  /// Set of writes that have been read on any path leading to this point
  llvm::BitVector Read;

  bool operator==(const RegisterWriters &Other) const = default;

  RegisterWriters &operator|=(const RegisterWriters &Other) {
    Reaching |= Other.Reaching;
    Read |= Other.Read;
    return *this;
  }

  RegisterWriters &operator&=(const RegisterWriters &Other) {
    Reaching &= Other.Reaching;
    Read &= Other.Read;
    return *this;
  }
};

/// One entry per register
class WritersSet : public llvm::SmallVector<RegisterWriters, 16> {
public:
  static WritersSet empty() { return {}; }

public:
  void dump() const debug_function {
    for (unsigned I = 0; I < size(); ++I) {
      dbg << I << ":\n";

      dbg << "  Reaching: ";
      if ((*this)[I].Reaching.size() != 0)
        for (const auto &Word : (*this)[I].Reaching.getData())
          dbg << " " << Word;
      dbg << "\n";

      dbg << "  Read: ";
      if ((*this)[I].Read.size() != 0)
        for (const auto &Word : (*this)[I].Read.getData())
          dbg << " " << Word;
      dbg << "\n";
    }
  }
};

class ReachingDefinitions {
public:
  using LatticeElement = WritersSet;
  using GraphType = Function *;
  using Label = BlockNode *;

private:
  llvm::DenseMap<const Operation *, uint8_t> WriteToIndex;
  WritersSet Default = LatticeElement::empty();

public:
  ReachingDefinitions(const Function &F) {
    // Populate WriteToIndex
    llvm::SmallVector<int> RegisterWriteIndex(F.registersCount(), 0);
    for (const Block *Block : F.nodes()) {
      for (const Operation &Operation : Block->Operations) {
        if (Operation.Type == OperationType::Write) {
          WriteToIndex[&Operation] = RegisterWriteIndex[Operation.Target];
          RegisterWriteIndex[Operation.Target] += 1;
        }
      }
    }

    Default.resize(RegisterWriteIndex.size());
    for (unsigned I = 0; I < RegisterWriteIndex.size(); ++I) {
      Default[I].Reaching.resize(RegisterWriteIndex[I]);
      Default[I].Read.resize(RegisterWriteIndex[I]);
    }
  }

public:
  static llvm::BitVector compute(const WritersSet &ProgramPoint,
                                 const WritersSet &Sink) {
    llvm::BitVector Result;
    revng_assert(Sink.size() == ProgramPoint.size());
    Result.resize(ProgramPoint.size());

    unsigned Index = 0;
    for (const auto &[AtPoint, AtSink] : zip(ProgramPoint, Sink)) {
      auto Unread = AtSink.Read;
      Unread.flip();
      Result[Index] = AtPoint.Reaching.anyCommon(Unread);
      ++Index;
    }

    return Result;
  }

public:
  WritersSet defaultValue() const { return Default; }

public:
  WritersSet combineValues(const WritersSet &LHS, const WritersSet &RHS) const {
    WritersSet Result = LHS;
    for (const auto &[ResultEntry, RHSEntry] : zip(Result, RHS)) {
      ResultEntry |= RHSEntry;
    }
    return Result;
  }

  bool isLessOrEqual(const WritersSet &LHS, const WritersSet &RHS) const {
    for (const auto &[LHSEntry, RHSEntry] : zip(LHS, RHS)) {
      auto Intersection = LHSEntry;
      Intersection &= RHSEntry;
      if (Intersection != LHSEntry)
        return false;
    }

    return true;
  }

  WritersSet applyTransferFunction(const Block *Block,
                                   const WritersSet &InitialState) const {
    WritersSet Result = InitialState;

    for (const Operation &Operation : *Block) {
      RegisterWriters &Writes = Result[Operation.Target];

      switch (Operation.Type) {
      case OperationType::Write: {
        Writes.Reaching.reset();
        Writes.Reaching.set(WriteToIndex.find(&Operation)->second);
      } break;
      case OperationType::Clobber: {
        Writes.Reaching.reset();
      } break;
      case OperationType::Read: {
        Writes.Read |= Writes.Reaching;
      } break;
      case OperationType::Invalid:
        revng_abort();
        break;
      }
    }

    return Result;
  }
};

static_assert(MFP::MonotoneFrameworkInstance<ReachingDefinitions>);

} // namespace rua
