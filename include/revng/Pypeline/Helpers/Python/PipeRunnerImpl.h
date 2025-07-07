#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "nanobind/nanobind.h"

#include "revng/Pypeline/Common.h"
#include "revng/Pypeline/Helpers/PipeRunner.h"
#include "revng/Pypeline/Helpers/Python/RunnerInfo.h"
#include "revng/Pypeline/Helpers/Python/Utils.h"
#include "revng/Pypeline/Model.h"

namespace revng::pypeline::helpers::python {

template<typename T>
class PipeRunnerImpl {
public:
  static ObjectDependencies run(T *Handle,
                                nanobind::handle_t<Model> TheModel,
                                nanobind::list Containers,
                                nanobind::list Incoming,
                                nanobind::list Outgoing,
                                nanobind::str Configuration) {
    const Model *CppModel = nanobind::cast<Model *>(TheModel);
    Request CppIncoming = convertRequests(Incoming);
    Request CppOutgoing = convertRequests(Outgoing);
    llvm::StringRef CppConfiguration(Configuration.c_str());

    return PipeRunner<RunnerInfo<T>>::run(*Handle,
                                          &T::run,
                                          CppModel,
                                          CppIncoming,
                                          CppOutgoing,
                                          CppConfiguration,
                                          Containers);
  }
};

} // namespace revng::pypeline::helpers::python
