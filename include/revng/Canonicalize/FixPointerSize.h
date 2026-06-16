#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/PipeboxCommon/Common.h"
#include "revng/PipeboxCommon/LLVMContainer.h"
#include "revng/PipeboxCommon/Model.h"

namespace revng::pypeline::pipes {

// For target binaries with 32-bit pointers, rewrite the LLVM module's
// DataLayout so the default-address-space pointer is 32 bits wide before the
// IR is handed off to `clifter`.
//
// Two preconditions must hold for the rewrite to be valid:
//
//   1. No scalar `alloca`'s allocated type transitively contains an
//      `llvm::PointerType`. Pointer-typed allocas would silently shrink
//      from 8 to 4 bytes when the DataLayout changes; pointer values must
//      instead live in integer-typed allocas and be bitcast (via
//      `ptrtoint`/`inttoptr`) at load/store boundaries.
//
//   2. No `getelementptr` has a source type that transitively contains an
//      `llvm::PointerType`. Otherwise the GEP's byte offsets would shift
//      under the new DataLayout.
//
// Both preconditions are checked with hard assertions; the pipe aborts if
// either fails. The DataLayout rewrite is a no-op for targets whose pointer
// is not 32 bits wide.
class FixPointerSize {
public:
  static constexpr llvm::StringRef Name = "fix-pointer-size";
  using Arguments = TypeList<PipeArgument<"Module",
                                          "LLVM Modules whose pointer size "
                                          "needs to match the "
                                          "target architecture">>;

  const std::string StaticConfiguration;
  FixPointerSize(llvm::StringRef StaticConfiguration) :
    StaticConfiguration(StaticConfiguration) {}

  PipeOutput run(const Model &TheModel,
                 const revng::pypeline::Request &Incoming,
                 const revng::pypeline::Request &Outgoing,
                 llvm::StringRef Configuration,
                 LLVMFunctionContainer &Container);
};

} // namespace revng::pypeline::pipes
