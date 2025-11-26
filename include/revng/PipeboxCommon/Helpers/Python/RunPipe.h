#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "nanobind/nanobind.h"

#include "revng/PipeboxCommon/Common.h"
#include "revng/PipeboxCommon/Helpers/PipeRunner.h"
#include "revng/PipeboxCommon/Helpers/Python/Helpers.h"
#include "revng/PipeboxCommon/Model.h"

namespace revng::pypeline::helpers::python {

template<typename T>
inline PipeOutput runPipe(T &Handle,
                          nanobind::object FileStorage,
                          nanobind::object TheModel,
                          nanobind::list Containers,
                          Request Incoming,
                          Request Outgoing,
                          llvm::StringRef Configuration) {
  const Model &CppModel = convertReadOnlyModel(TheModel);
  auto ContainerTuple = containerVectorToTuple<T>(Containers);

  {
    nanobind::gil_scoped_release X;
    return revng::pypeline::helpers::runPipe(Handle,
                                             CppModel,
                                             Incoming,
                                             Outgoing,
                                             Configuration,
                                             ContainerTuple);
  }
}

} // namespace revng::pypeline::helpers::python
