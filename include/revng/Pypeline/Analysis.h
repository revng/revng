#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/StringRef.h"

#include "revng/Pypeline/Concepts.h"
#include "revng/Pypeline/ModelImpl.h"
#include "revng/Pypeline/Registry.h"
#include "revng/Pypeline/Utils.h"
#include "revng/Support/Assert.h"

namespace detail {

template<typename T>
struct NanobindAnalysisRunner {
  template<size_t... I>
  using integer_sequence = std::integer_sequence<size_t, I...>;

  static bool run(T *Handle,
                  nanobind::handle_t<Model> TheModel,
                  nanobind::list Containers,
                  nanobind::list Incoming,
                  nanobind::str Configuration) {
    Model *CppModel = nanobind::cast<Model *>(TheModel);
    detail::RequestT CppIncoming = detail::convertRequests(Incoming);
    llvm::StringRef CppConfiguration(Configuration.c_str());

    return AnalysisRunner<NanobindRunnerInfo<T>>::runImpl(*Handle,
                                                          &T::run,
                                                          CppModel,
                                                          CppIncoming,
                                                          CppConfiguration,
                                                          Containers);
  }
};

} // namespace detail

template<IsAnalysis T>
struct RegisterAnalysis {
  RegisterAnalysis() {
    TheRegistry.registerPythonCallback([](nanobind::module_ &M) {
      nanobind::object BaseClass = detail::getBaseClass("Analysis");
      nanobind::class_<T>(M, T::Name.data(), BaseClass)
        .def(nanobind::init<>())
        .def("run", &detail::NanobindAnalysisRunner<T>::run);
    });
  }
};
