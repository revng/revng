/// \file OriginalAssemblyAnnotationWriter.cpp
/// This file handles debugging information generation.

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instruction.h"
#include "llvm/Support/FormattedStream.h"

#include "revng/ADT/STLExtras.h"
#include "revng/Model/FunctionTags.h"
#include "revng/Recompile/OriginalAssemblyAnnotationWriter.h"
#include "revng/Support/IRHelpers.h"

using namespace llvm;

/// Writes the text contained in the metadata with the specified kind ID to the
/// output stream, unless that metadata is exactly the same as in the previous
/// instruction.
static void writeMetadataIfNew(const Instruction *I,
                               unsigned MDKind,
                               formatted_raw_ostream &Output,
                               StringRef Prefix) {
  auto BeginIt = I->getParent()->begin();
  StringRef Text = getText(I, MDKind);
  if (Text.size()) {
    StringRef LastText;

    do {
      if (I->getIterator() == BeginIt) {
        I = nullptr;
      } else {
        I = I->getPrevNode();
        LastText = getText(I, MDKind);
      }
    } while (I != nullptr && LastText.size() == 0);

    if (I == nullptr or LastText != Text) {
      std::string TextToSerialize = Text.str();
      replaceAll(TextToSerialize, "\n", " ");
      Output << Prefix.data() << TextToSerialize << "\n";
    }
  }
}

using OAAW = OriginalAssemblyAnnotationWriter;
void OAAW::emitInstructionAnnot(const Instruction *I,
                                formatted_raw_ostream &Output) {

  // Ignore whatever is outside the root and the isolated functions
  writeMetadataIfNew(I, PTCInstrMDKind, Output, "\n  ; ");
  if (isRootOrLifted(I->getParent()->getParent())) {
  }
}
