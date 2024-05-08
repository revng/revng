#pragma once

//
// This file is distributed under the MIT License. See LICENSE.mit for details.
//

#include <cmath>
#include <map>
#include <string>

#include "llvm/IR/Function.h"

#include "revng/Support/Debug.h"

namespace FunctionTags {
class Tag;
}

class ListStatistics {
private:
  unsigned ElementsCount = 0;
  unsigned Sum = 0;
  double OldM = 0.0;
  double NewM = 0.0;
  double OldS = 0.0;
  double NewS = 0.0;

public:
  unsigned count() const { return ElementsCount; }
  unsigned sum() const { return Sum; }
  double mean() const { return (count() > 0) ? NewM : 0.0; }
  double variance() const {
    return ((ElementsCount > 1) ? NewS / (ElementsCount - 1) : 0.0);
  }
  double standardDeviation() const { return sqrt(variance()); }
  double confidenceInterval() const {
    return 1.96 * standardDeviation() / sqrt(ElementsCount);
  }

public:
  std::string toString() const {
    std::string Result;
    Result += std::to_string(unsigned(mean()));
    Result += "±";
    Result += std::to_string(unsigned(confidenceInterval()));
    return Result;
  }

public:
  void push(unsigned NewElement) {
    ++ElementsCount;
    Sum += NewElement;

    // See Knuth TAOCP vol 2, 3rd edition, page 232
    if (ElementsCount == 1) {
      OldM = NewM = NewElement;
      OldS = 0.0;
    } else {
      NewM = OldM + (NewElement - OldM) / ElementsCount;
      NewS = OldS + (NewElement - OldM) * (NewElement - NewM);

      // Set up for next iteration
      OldM = NewM;
      OldS = NewS;
    }
  }
};

class FunctionClass {
public:
  unsigned DeclarationsCount = 0;
  unsigned DefinitionsCount = 0;
  ListStatistics InstructionsStatistics;

public:
  void process(const llvm::Function &F) {
    if (F.isDeclaration()) {
      ++DeclarationsCount;
    } else {
      ++DefinitionsCount;

      unsigned Size = 0;
      for (const llvm::BasicBlock &BB : F)
        Size += BB.size();
      InstructionsStatistics.push(Size);
    }
  }

  void dump(llvm::raw_ostream &Output,
            unsigned Indent,
            const FunctionClass *Old = nullptr) const;
};

class ModuleStatistics {
private:
  unsigned NamedGlobalsCount = 0;
  unsigned AnonymousGlobalsCount = 0;
  unsigned AliasesCount = 0;

  unsigned NamedMetadataCount = 0;
  unsigned MaxNamedMetadataSize = 0;

  unsigned MaxArrayElements = 0;
  unsigned MaxStructElements = 0;

  FunctionClass AllFunctions;
  std::map<const FunctionTags::Tag *, FunctionClass> TaggedFunctions;

  unsigned NamedStructsCount = 0;
  unsigned AnonymousStructsCount = 0;

  unsigned DebugCompileUnitsCount = 0;
  unsigned DebugScopesCount = 0;
  unsigned DebugSubprogramsCount = 0;
  unsigned DebugGloblaVariablesCount = 0;
  unsigned DebugTypesCount = 0;

public:
  static ModuleStatistics analyze(const llvm::Module &M);

  void dump() const debug_function {
    std::string Result;
    {
      llvm::raw_string_ostream S(Result);
      dump(S, 0, nullptr);
    }
    dbg << Result;
  }

  void dump(llvm::raw_ostream &Output,
            unsigned Indent,
            const ModuleStatistics *Old = nullptr) const;
};
