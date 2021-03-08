/// \file ABIDetectionPass.cpp
/// \brief

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <fstream>

#include "llvm/Support/CommandLine.h"

#include "revng/StackAnalysis/ABIDetectionPass.h"

using namespace llvm;
using namespace llvm::cl;

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

static void serializeFunctionBoundaries(std::ostream &Output, Module &M) {
  using namespace llvm;

  QuickMetadata QMD(getContext(&M));
  Function &F = *M.getFunction("root");
  std::map<StringRef, std::vector<BasicBlock *>> Functions;

  for (BasicBlock &BB : F) {
    if (!BB.empty()) {
      Instruction *Terminator = BB.getTerminator();
      if (MDNode *Node = Terminator->getMetadata("revng.func.member.of")) {
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

namespace StackAnalysis {

char ABIDetectionPass::ID = 0;
using Register = RegisterPass<ABIDetectionPass>;
static Register X("detect-abi", "ABI Detection Pass", true, true);

bool ABIDetectionPass::runOnModule(Module &M) {
  auto &GCBI = getAnalysis<GeneratedCodeBasicInfoWrapperPass>().getGCBI();
  auto &SA = getAnalysis<StackAnalysis>();
  SA.serializeMetadata(*M.getFunction("root"), GCBI);

  if (FBDPOutputPath.getNumOccurrences() == 1) {
    std::ofstream Output;
    serializeFunctionBoundaries(pathToStream(FBDPOutputPath, Output), M);
  }

  return false;
}

} // namespace StackAnalysis
