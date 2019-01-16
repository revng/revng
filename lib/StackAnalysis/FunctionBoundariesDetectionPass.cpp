/// \file FunctionBoundariesDetectionPass.cpp
/// \brief

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// Standard includes
#include <fstream>

// Local libraries includes
#include "revng/StackAnalysis/FunctionBoundariesDetectionPass.h"
#include "revng/Support/CommandLine.h"

using namespace llvm;
using namespace llvm::cl;

namespace StackAnalysis {

using FBDP = FunctionBoundariesDetectionPass;

char FBDP::ID = 0;
using Register = RegisterPass<FunctionBoundariesDetectionPass>;
static Register X("detect-function-boundaries",
                  "Function Boundaries Detection Pass",
                  true,
                  true);

static opt<std::string> FBDPOutputPath("detect-function-boundaries-output",
                                       desc("Destination path for the Function "
                                            "Boundaries Detection Pass"),
                                       value_desc("path"),
                                       cat(MainCategory));

template<typename T>
struct CompareByName {
  bool operator()(const T *LHS, const T *RHS) const {
    return LHS->getName() < RHS->getName();
  }
};

void FBDP::serialize(std::ostream &Output, Module &M) {
  QuickMetadata QMD(getContext(&M));
  Function &F = *M.getFunction("root");
  std::map<llvm::StringRef, std::vector<llvm::BasicBlock *>> Functions;

  for (BasicBlock &BB : F) {
    if (!BB.empty()) {
      TerminatorInst *Terminator = BB.getTerminator();
      if (MDNode *Node = Terminator->getMetadata("func.member.of")) {
        auto *Tuple = cast<MDTuple>(Node);
        for (const MDOperand &Op : Tuple->operands()) {
          auto *FunctionMD = cast<MDTuple>(Op);
          auto *FirstOperand = QMD.extract<MDTuple *>(FunctionMD, 0);
          auto *FunctionNameMD = QMD.extract<MDString *>(FirstOperand, 0);
          Functions[FunctionNameMD->getString()].push_back(&BB);
        }
      }
    }
  }

  Output << "function,basicblock\n";

  auto Comparator = CompareByName<const BasicBlock>();
  for (auto &P : Functions) {
    std::sort(P.second.begin(), P.second.end(), Comparator);
    for (BasicBlock *BB : P.second)
      Output << P.first.data() << "," << BB->getName().data() << "\n";
  }
}

bool FBDP::runOnModule(Module &M) {
  auto &SA = getAnalysis<StackAnalysis<false>>();
  SA.serializeMetadata(*M.getFunction("root"));

  if (FBDPOutputPath.getNumOccurrences() == 1) {
    std::ofstream Output;
    serialize(pathToStream(FBDPOutputPath, Output), M);
  }

  return false;
}

} // namespace StackAnalysis
