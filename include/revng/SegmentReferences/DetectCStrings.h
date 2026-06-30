#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Model/GlobalVariableBuilder.h"
#include "revng/PipeboxCommon/BinariesContainer.h"
#include "revng/PipeboxCommon/Common.h"
#include "revng/PipeboxCommon/LLVMContainer.h"
#include "revng/PipeboxCommon/Model.h"
#include "revng/SegmentReferences/SegmentUsesEnumerator.h"

class RawBinaryView;

/// Analyze segment references in the code, detect pointers to global strings
/// and add corresponding fields in the struct describing segments
class DetectCStrings {
private:
  SegmentUsesEnumerator SegmentUses;
  RawBinaryView &BinaryView;
  model::GlobalVariableBuilder GlobalBuilder;

public:
  DetectCStrings(model::Binary &Binary, RawBinaryView &BinaryView) :
    SegmentUses(Binary, SegmentUsesEnumerator::SegmentAccess::ReadOnly),
    BinaryView(BinaryView),
    GlobalBuilder(Binary) {}

  void run(llvm::Module &M, llvm::Function *LimitTo = nullptr);
};

namespace revng::pypeline::analyses {

class DetectCStrings {
public:
  static constexpr llvm::StringRef Name = "detect-c-strings";

  llvm::Error run(Model &Model,
                  const Request &Incoming,
                  llvm::StringRef Configuration,
                  const BinariesContainer &Binaries,
                  LLVMFunctionContainer &ModuleContainer);
};

} // namespace revng::pypeline::analyses
