/// \file ApplyDiffAnalysis.cpp
/// \brief Pipes contains all the various pipes and kinds exposed by revng

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/Support/Error.h"

#include "revng/Pipeline/Context.h"
#include "revng/Pipeline/RegisterAnalysis.h"
#include "revng/Pipes/ApplyDiffAnalysis.h"
#include "revng/Pipes/ModelGlobal.h"
#include "revng/Support/YAMLTraits.h"
#include "revng/TupleTree/TupleTreeDiff.h"

namespace revng::pipes {

llvm::Error ApplyDiffAnalysis::run(pipeline::Context &Ctx,
                                   std::string DiffLocation) {
  if (DiffLocation == "")
    return llvm::Error::success();

  auto &Model = getWritableModelFromContext(Ctx);

  using DiffT = TupleTreeDiff<model::Binary>;

  auto MaybeDiff = deserializeFileOrSTDIN<DiffT>(DiffLocation);
  if (!MaybeDiff)
    return MaybeDiff.takeError();

  return MaybeDiff->apply(Model);
}

static pipeline::RegisterAnalysis<ApplyDiffAnalysis> X;

} // namespace revng::pipes
