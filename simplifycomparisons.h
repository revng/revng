#ifndef _SIMPLIFYCOMPARISON_H
#define _SIMPLIFYCOMPARISON_H

// Standard includes
#include <unordered_map>

// LLVM includes
#include "llvm/Pass.h"

// Local includes
#include "reachingdefinitions.h"

/// \brief Look for sophisticated comparisons that can be simplified
/// This pass looks for comparisons checkin for the sign of a value, and, if
/// possible, tries to build the boolean expression used to compute that value.
/// If all the expression depends only on the operands of a subtraction
/// operation, the table of truth of that boolean expressions is built, and is
/// matched against a known list. In case it corresponds to a known expression,
/// it is replaced with associated, simpler comparison.
/// This is particularly useful on ARM.
class SimplifyComparisonsPass : public llvm::FunctionPass {
public:
  static char ID;

public:
  struct Comparison {
    Comparison() { }
    Comparison(llvm::CmpInst *Cmp) :
      Predicate(Cmp->getPredicate()),
      LHS(Cmp->getOperand(0)),
      RHS(Cmp->getOperand(1)) { }

    llvm::CmpInst::Predicate Predicate;
    llvm::Value *LHS;
    llvm::Value *RHS;
  };

public:

  SimplifyComparisonsPass() : llvm::FunctionPass(ID) { }

  void getAnalysisUsage(llvm::AnalysisUsage &AU) const override {
    AU.setPreservesAll();
    AU.addRequired<ReachingDefinitionsPass>();
  }

  // TODO: this pass ignores function calls, and in particular calls to newpc,
  //       which could lead to a corrupted IR in case of split between the
  //       instruction generating the condition (e.g. ARM's cmp) and its user
  //       (e.g. blt instruction).
  bool runOnFunction(llvm::Function &F) override;

  llvm::Value *findOldest(llvm::Value *V);
  Comparison getComparison(llvm::CmpInst *Cmp) {
    auto It = SimplifiedComparisons.find(Cmp);
    if (It != SimplifiedComparisons.end())
      return It->second;
    else
      return Comparison(Cmp);
  }

  virtual void releaseMemory() override {
    DBG("release", { dbg << "SimplifyComparisonsPass is releasing memory\n"; });
    freeContainer(SimplifiedComparisons);
  }

private:
  ReachingDefinitionsPass *RDP;
  std::unordered_map<llvm::CmpInst *, Comparison> SimplifiedComparisons;
};

#endif // _SIMPLIFYCOMPARISON_H
