#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "nanobind/nanobind.h"
#include "nanobind/stl/optional.h"
#include "nanobind/stl/pair.h"
#include "nanobind/stl/set.h"
#include "nanobind/stl/string.h"
#include "nanobind/stl/vector.h"

#include "revng/Pypeline/Concepts.h"
#include "revng/Pypeline/Helpers/Python/AnalysisRunnerImpl.h"
#include "revng/Pypeline/Helpers/Python/Casters.h"
#include "revng/Pypeline/Helpers/Python/ContainerIO.h"
#include "revng/Pypeline/Helpers/Python/PipeRunnerImpl.h"
#include "revng/Pypeline/Helpers/Python/Registry.h"

template<IsAnalysis T>
struct RegisterAnalysis {
  RegisterAnalysis() {
    using namespace revng::pypeline::helpers;

    // Python
    python::Registry.registerModuleInitializer([](nanobind::module_ &M,
                                                  python::BaseClasses &BC) {
      nanobind::class_<T>(M, T::Name.data(), BC.BaseAnalysis)
        .def(nanobind::init<>())
        .def("run", &python::AnalysisRunnerImpl<T>::run);
    });
  }
};

template<IsContainer T>
struct RegisterContainer {
  RegisterContainer() {
    using namespace revng::pypeline::helpers;

    // Python
    python::Registry.registerModuleInitializer([](nanobind::module_ &M,
                                                  python::BaseClasses &BC) {
      nanobind::class_<T>(M, T::Name.data(), BC.BaseContainer)
        .def_ro_static("kind", &T::Kind)
        .def(nanobind::init<>())
        .def("objects", &T::objects)
        .def("verify", &T::verify)
        .def("deserialize", &python::ContainerIO<T>::deserialize)
        .def("serialize", &python::ContainerIO<T>::serialize);
    });
  }
};

template<IsPipe T>
struct RegisterPipe {
  RegisterPipe() {
    using namespace revng::pypeline::helpers;

    // Python
    python::Registry.registerModuleInitializer([](nanobind::module_ &M,
                                                  python::BaseClasses &BC) {
      nanobind::class_<T>(M, T::Name.data(), BC.BasePipe)
        .def(nanobind::init<llvm::StringRef>())
        .def("run", &python::PipeRunnerImpl<T>::run);
    });
  }
};
