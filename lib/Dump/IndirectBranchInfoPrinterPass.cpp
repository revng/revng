/// \file IndirectBranchInfoPrinterPass.cpp
/// \brief Serialize the results of the StackAnalysis on disk.

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/InstrTypes.h"

#include "revng/Dump/IndirectBranchInfoPrinterPass.h"
#include "revng/Support/IRHelpers.h"

void IndirectBranchInfoPrinterPass::serialize(
  llvm::raw_ostream &OS,
  llvm::CallBase *Call,
  llvm::SmallVectorImpl<llvm::GlobalVariable *> &ABIRegs) {
  auto Text = llvm::MemoryBuffer::getFileAsStream(OutputFile);
  if (!Text.getError()) {
    if (!Text->get()->getBuffer().startswith("name")) {
      OS << "name,ra,fso,address";
      for (const auto &Reg : ABIRegs)
        OS << "," << Reg->getName();
      OS << "\n";
    }
  }

  OS << Call->getParent()->getParent()->getName();
  for (unsigned I = 0; I < Call->getNumArgOperands(); ++I) {
    if (isa<llvm::ConstantInt>(Call->getArgOperand(I)))
      OS << ","
         << cast<llvm::ConstantInt>(Call->getArgOperand(I))->getSExtValue();
    else
      OS << ",unknown";
  }
  OS << "\n";
}

llvm::PreservedAnalyses
IndirectBranchInfoPrinterPass::run(llvm::Function &F,
                                   llvm::FunctionAnalysisManager &FAM) {
  auto &M = *F.getParent();
  auto &MAM = FAM.getResult<llvm::ModuleAnalysisManagerFunctionProxy>(F)
                .getManager();
  GCBI = MAM.getCachedResult<GeneratedCodeBasicInfoAnalysis>(M);
  if (!GCBI)
    GCBI = &(FAM.getResult<GeneratedCodeBasicInfoAnalysis>(F));
  revng_assert(GCBI != nullptr);

  llvm::SmallVector<llvm::GlobalVariable *, 16> ABIRegisters;
  for (auto *CSV : GCBI->abiRegisters())
    if (CSV && !(GCBI->isSPReg(CSV)))
      ABIRegisters.emplace_back(CSV);

  std::error_code EC;
  llvm::raw_fd_ostream OS(OutputFile,
                          EC,
                          llvm::sys::fs::OF_Append | llvm::sys::fs::OF_Text);
  if (!EC) {
    for (auto *Call : callers(M.getFunction("indirect_branch_info")))
      if (Call->getParent()->getParent() == &F)
        serialize(OS, Call, ABIRegisters);
    OS.close();
  }

  return llvm::PreservedAnalyses::all();
}
