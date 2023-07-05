/// \file AAWriterPass.cpp
/// Add aliasing scope information to the IR before load and store accesses.

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <vector>

#include "llvm/IR/Metadata.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FormattedStream.h"

#include "revng/EarlyFunctionAnalysis/AAWriterPass.h"
#include "revng/Support/IRHelpers.h"

using namespace llvm;

class AliasAnalysisAnnotatedWriter : public AssemblyAnnotationWriter {
  const bool StoresOnly;

public:
  AliasAnalysisAnnotatedWriter(bool StoresOnly) : StoresOnly(StoresOnly) {}

  void emitInstructionAnnot(const Instruction *I,
                            formatted_raw_ostream &) override;
};

using AAAW = AliasAnalysisAnnotatedWriter;
void AAAW::emitInstructionAnnot(const Instruction *I,
                                formatted_raw_ostream &OS) {
  QuickMetadata QMD(I->getContext());
  if (!isa<StoreInst>(I) || (!StoresOnly && !isa<LoadInst>(I)))
    return;

  OS << "\n";
  OS.PadToColumn(2);
  OS << "; alias.scope: ";
  auto *AliasScopeMD = I->getMetadata(LLVMContext::MD_alias_scope);
  for (unsigned I = 0, E = AliasScopeMD->getNumOperands(); I != E; ++I) {
    auto *Tuple = cast<MDTuple>(AliasScopeMD->getOperand(I));
    OS << QMD.extract<StringRef>(Tuple, 0);
    if (I != (E - 1))
      OS << ", ";
  }

  OS << "\n";
  OS.PadToColumn(2);
  OS << "; noalias: ";
  auto *NoAliasScopeMD = I->getMetadata(LLVMContext::MD_noalias);
  for (unsigned I = 0, E = NoAliasScopeMD->getNumOperands(); I != E; ++I) {
    auto *Tuple = cast<MDTuple>(NoAliasScopeMD->getOperand(I));
    OS << QMD.extract<StringRef>(Tuple, 0);
    if (I != (E - 1))
      OS << ", ";
  }
  OS << "\n";
}

PreservedAnalyses AAWriterPass::run(Function &F, FunctionAnalysisManager &FAM) {
  std::unique_ptr<llvm::AssemblyAnnotationWriter> Annotator;
  Annotator.reset(new AliasAnalysisAnnotatedWriter(StoresOnly));

  formatted_raw_ostream FOS(OS);
  F.print(OS, Annotator.get(), true);

  return PreservedAnalyses::all();
}
