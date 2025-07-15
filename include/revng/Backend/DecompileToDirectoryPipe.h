#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <array>
#include <string>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/raw_ostream.h"

#include "revng/EarlyFunctionAnalysis/CFGStringMap.h"
#include "revng/Pipeline/Contract.h"
#include "revng/Pipes/FileContainer.h"
#include "revng/Pipes/Kinds.h"

namespace revng::pipes {

inline constexpr char RecompilableArchiveMime[] = "application/"
                                                  "x.recompilable-archive"
                                                  "+tar+gz";
inline constexpr char RecompilableArchiveName[] = "recompilable-archive";
inline constexpr char RecompilableArchiveExtension[] = ".tar.gz";
using RecompilableArchiveContainer = FileContainer<
  &kinds::RecompilableArchive,
  RecompilableArchiveName,
  RecompilableArchiveMime,
  RecompilableArchiveExtension>;

class DecompileToDirectory {
public:
  static constexpr auto Name = "decompile-to-directory";

  std::array<pipeline::ContractGroup, 1> getContract() const {
    using namespace pipeline;
    using namespace revng::kinds;

    return { ContractGroup({ Contract(StackAccessesSegregated,
                                      0,
                                      RecompilableArchive,
                                      2,
                                      InputPreservation::Preserve),
                             Contract(CFG,
                                      1,
                                      RecompilableArchive,
                                      2,
                                      InputPreservation::Preserve) }) };
  }

  void run(pipeline::ExecutionContext &EC,
           pipeline::LLVMContainer &IRContainer,
           const revng::pipes::CFGMap &CFGMap,
           RecompilableArchiveContainer &OutTarFile);
};

} // end namespace revng::pipes
