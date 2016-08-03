#ifndef _SIMPLIFYCOMPARISON_H
#define _SIMPLIFYCOMPARISON_H

// LLVM includes
#include "llvm/Pass.h"

/// \brief Look for sophisticated comparisons that can be simplified
/// This pass looks for comparisons checkin for the sign of a value, and, if
/// possible, tries to build the boolean expression used to compute that value.
/// If all the expression depends only on the operands of a subtraction
/// operation, the table of truth of that boolean expressions is built, and is
/// matched against a known list. In case it corresponds to a known expression,
/// it is replaced with associated, simpler comparison.
/// This is particularly useful on ARM.
class SimplifyComparisonPass : public llvm::FunctionPass {
public:
  static char ID;

  SimplifyComparisonPass() : llvm::FunctionPass(ID) { }

  // TODO: this pass ignores function calls, and in particular calls to newpc,
  //       which could lead to a corrupted IR in case of split between the
  //       instruction generating the condition (e.g. ARM's cmp) and its user
  //       (e.g. blt instruction).
  bool runOnFunction(llvm::Function &F) override;
};

#endif // _SIMPLIFYCOMPARISON_H
