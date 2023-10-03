//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/GraphTraits.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Support/Casting.h"

#include "revng/MFP/MFP.h"

#include "Analyses.h"

namespace ABIAnalyses::DeadRegisterArgumentsOfFunction {
using namespace llvm;
using namespace ABIAnalyses;

std::map<const GlobalVariable *, State>
analyze(const BasicBlock *FunctionEntry, const GeneratedCodeBasicInfo &GCBI) {
  using MFI = MFIAnalysis<true, CoreLattice>;

  MFI Instance{ { GCBI } };
  MFI::LatticeElement InitialValue;
  MFI::LatticeElement ExtremalValue(CoreLattice::ExtremalLatticeElement);

  auto
    Res = MFP::getMaximalFixedPoint<MFI, MFI::GT, MFI::LGT>(Instance,
                                                            FunctionEntry,
                                                            InitialValue,
                                                            ExtremalValue,
                                                            { FunctionEntry },
                                                            { FunctionEntry });

  std::map<const GlobalVariable *, State> RegNoOrDead{};

  for (auto &[BB, Result] : Res)
    for (auto &[GV, RegState] : Result.OutValue)
      if (RegState == CoreLattice::NoOrDead)
        RegNoOrDead[GV] = State::NoOrDead;

  return RegNoOrDead;
}
} // namespace ABIAnalyses::DeadRegisterArgumentsOfFunction
