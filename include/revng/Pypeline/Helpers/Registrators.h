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
#include "revng/Pypeline/Helpers/Native/AnalysisImpl.h"
#include "revng/Pypeline/Helpers/Native/ContainerImpl.h"
#include "revng/Pypeline/Helpers/Native/PipeImpl.h"
#include "revng/Pypeline/Helpers/Native/Registry.h"
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

    // Native
    revng_assert(native::Registry.Analyses.count(T::Name) == 0);
    native::Registry.Analyses[T::Name] =
      []() -> std::unique_ptr<native::Analysis> {
      return std::make_unique<native::AnalysisImpl<T>>();
    };
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

    // Native
    revng_assert(native::Registry.Containers.count(T::Name) == 0);
    native::Registry.Containers[T::Name] =
      []() -> std::unique_ptr<native::Container> {
      return std::make_unique<native::ContainerImpl<T>>();
    };
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

    // Native
    revng_assert(native::Registry.Pipes.count(T::Name) == 0);
    native::Registry.Pipes[T::Name] =
      [](llvm::StringRef Config) -> std::unique_ptr<native::Pipe> {
      return std::make_unique<native::PipeImpl<T>>(Config);
    };
  }
};
