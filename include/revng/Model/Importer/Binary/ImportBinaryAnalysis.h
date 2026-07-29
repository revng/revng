#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/raw_ostream.h"

#include "revng/PipeboxCommon/BinariesContainer.h"
#include "revng/PipeboxCommon/Common.h"

namespace revng::pypeline::analyses {

/// This Analysis, given the BinariesContainer containing the input binaries,
/// will import the segments of the binary (by reading the ELF/Mach-O/PECOFF)
/// and try and import debugging information for it and any library it uses.
class ParseBinaryAnalysis {
public:
  static constexpr llvm::StringRef Name = "parse-binary";

  llvm::Error run(Model &Model,
                  const Request &Incoming,
                  llvm::StringRef Configuration,
                  const BinariesContainer &Binaries);
};

} // namespace revng::pypeline::analyses
