#pragma once

//
// Copyright rev.ng Srls. See LICENSE.md for details.
//
#include <map>

#include "llvm/IR/Value.h"
#include "llvm/Pass.h"

#include "revng-c/ValueManipulationAnalysis/TypeColors.h"

/// Assign a type color for each Use and Value of a function
class ValueManipulationAnalysis : public llvm::FunctionPass {
public:
  using ColorMapT = std::map<const llvm::Value *, const vma::ColorSet>;
  static char ID;

  // Ctors
  ValueManipulationAnalysis() : llvm::FunctionPass(ID) {}

  // FunctionPass methods
  bool runOnFunction(llvm::Function &F) override;
  void getAnalysisUsage(llvm::AnalysisUsage &AU) const override;

  /// ColorMap getter
  ///\return A map between `llvm::Value*`s and their type color
  const ColorMapT &getColorMap() const { return ColorMap; }

private:
  ColorMapT ColorMap;
};
