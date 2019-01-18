/// \file collectnoreturn.cpp
/// \brief Implementation of the pass to collect the list of noreturn basic
///        blocks

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// LLVM includes
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"

// Local libraries includes
#include "revng/Dump/CollectNoreturn.h"
#include "revng/Support/CommandLine.h"

using namespace llvm;
using namespace llvm::cl;

template<typename T>
struct CompareByName {
  bool operator()(const T *LHS, const T *RHS) const {
    return LHS->getName() < RHS->getName();
  }
};

char CollectNoreturn::ID = 0;
using RegisterCNR = RegisterPass<CollectNoreturn>;
static RegisterCNR X("collect-noreturn", "Collect noreturn Pass", true, true);

static opt<std::string> OutputPath("collect-noreturn-output",
                                   desc("Destination path for the Collect "
                                        "noreturn Pass"),
                                   value_desc("path"),
                                   cat(MainCategory));

void CollectNoreturn::serialize(std::ostream &Output) {
  Output << "noreturn\n";
  for (BasicBlock *BB : NoreturnBBs)
    Output << BB->getName().data() << "\n";
}

bool CollectNoreturn::runOnModule(Module &M) {
  Function &F = *M.getFunction("root");
  NoreturnBBs.clear();

  for (BasicBlock &BB : F) {
    if (!BB.empty()) {
      TerminatorInst *Terminator = BB.getTerminator();
      if (Terminator->getMetadata("noreturn") != nullptr)
        NoreturnBBs.push_back(&BB);
    }
  }

  std::sort(NoreturnBBs.begin(),
            NoreturnBBs.end(),
            CompareByName<BasicBlock>());

  if (OutputPath.getNumOccurrences() == 1) {
    std::ofstream Output;
    serialize(pathToStream(OutputPath, Output));
  }

  return false;
}
