#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/IR/AssemblyAnnotationWriter.h"
#include "llvm/IR/LLVMContext.h"

/// AssemblyAnnotationWriter decorating the output original assembly/PTC
class OriginalAssemblyAnnotationWriter : public llvm::AssemblyAnnotationWriter {
public:
  OriginalAssemblyAnnotationWriter(llvm::LLVMContext &Context) :
    PTCInstrMDKind(Context.getMDKindID("pi")) {}

  ~OriginalAssemblyAnnotationWriter() override = default;

  virtual void
  emitInstructionAnnot(const llvm::Instruction *I,
                       llvm::formatted_raw_ostream &Output) override;

private:
  unsigned PTCInstrMDKind;
};
