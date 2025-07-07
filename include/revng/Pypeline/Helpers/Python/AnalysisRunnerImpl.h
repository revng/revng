#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "nanobind/nanobind.h"

#include "revng/Pypeline/Helpers/AnalysisRunner.h"
#include "revng/Pypeline/Helpers/Python/RunnerInfo.h"
#include "revng/Pypeline/Helpers/Python/Utils.h"
#include "revng/Pypeline/Model.h"

namespace revng::pypeline::helpers::python {

template<typename T>
struct AnalysisRunnerImpl {
  static llvm::Error run(T *Handle,
                         nanobind::handle_t<Model> TheModel,
                         nanobind::list Containers,
                         nanobind::list Incoming,
                         nanobind::str Configuration) {
    using namespace revng::pypeline::helpers::python;

    Model *CppModel = nanobind::cast<Model *>(TheModel);
    Request CppIncoming = convertRequests(Incoming);
    llvm::StringRef CppConfiguration(Configuration.c_str());

    return AnalysisRunner<RunnerInfo<T>>::run(*Handle,
                                              &T::run,
                                              CppModel,
                                              CppIncoming,
                                              CppConfiguration,
                                              Containers);
  }
};

} // namespace revng::pypeline::helpers::python
