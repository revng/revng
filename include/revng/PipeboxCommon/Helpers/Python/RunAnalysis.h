#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "nanobind/nanobind.h"

#include "revng/PipeboxCommon/Helpers/AnalysisRunner.h"
#include "revng/PipeboxCommon/Helpers/Python/Helpers.h"
#include "revng/PipeboxCommon/Model.h"

namespace revng::pypeline::helpers::python {

template<typename T>
inline llvm::Error runAnalysis(T &Handle,
                               nanobind::handle_t<Model> TheModel,
                               nanobind::list Containers,
                               Request Incoming,
                               llvm::StringRef Configuration) {
  Model *CppModel = nanobind::cast<Model *>(TheModel);
  auto ContainerTuple = containerVectorToTuple<T>(Containers);

  {
    nanobind::gil_scoped_release X;
    return revng::pypeline::helpers::runAnalysis(Handle,
                                                 *CppModel,
                                                 Incoming,
                                                 Configuration,
                                                 ContainerTuple);
  }
}

} // namespace revng::pypeline::helpers::python
