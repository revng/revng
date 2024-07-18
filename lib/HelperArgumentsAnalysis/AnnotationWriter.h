#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/IR/AssemblyAnnotationWriter.h"
#include "llvm/Support/FormattedStream.h"

#include "ArgumentUsageAnalysis.h"
#include "CPUStateUsage.h"

namespace aua {

class AnnotationWriter : public llvm::AssemblyAnnotationWriter {
private:
  ArgumentUsageAnalysis &AUA;
  CPUStateUsageAnalysis &CSUA;
  const Function *CurrentAUA = nullptr;

public:
  AnnotationWriter(ArgumentUsageAnalysis &AUA, CPUStateUsageAnalysis &CSUA) :
    AUA(AUA), CSUA(CSUA) {}
  ~AnnotationWriter() override = default;

public:
  virtual void emitFunctionAnnot(const llvm::Function *F,
                                 llvm::formatted_raw_ostream &Output) override {
    auto It = AUA.find(F);
    if (It == AUA.end()) {
      CurrentAUA = nullptr;
    } else {
      const Function &Results = It->second;
      CurrentAUA = &Results;

      if (auto *Results = CSUA.get(*const_cast<llvm::Function *>(F)))
        Results->dump(Output, "; ");

      Results.dump(Output, "; ");
    }
  }

  virtual void
  emitInstructionAnnot(const llvm::Instruction *I,
                       llvm::formatted_raw_ostream &Output) override {
    if (CurrentAUA != nullptr) {
      if (const Value *V = CurrentAUA->tryGet(*I)) {
        Output << "  ; " << V->toString() << "\n";
      }

      if (CSUA.isEscaping(*I))
        Output << "  ; CPU state escapes!\n";

      for (unsigned J = 0; J < I->getNumOperands(); ++J) {
        const auto &Accesses = CSUA.getOffsets(I->getOperandUse(J));
        if (Accesses.size() > 0) {
          Output << "  ; Offsets for operand " << J << ": {";
          for (const auto &[Offset, Size] : Accesses)
            Output << " i" << (Size * 8) << " @ " << Offset;
          Output << " }\n";
        }
      }

      if (auto *Call = dyn_cast<llvm::CallInst>(I)) {
        for (const auto &Call : CurrentAUA->calls()) {
          if (&Call.callInstruction() == I) {
            Call.dump(Output, "  ; ", true);
          }
        }
      }
    }
  }
};

} // namespace aua
