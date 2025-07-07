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

#include "revng/PipeboxCommon/Concepts.h"
#include "revng/PipeboxCommon/Helpers/Python/Casters.h"
#include "revng/PipeboxCommon/Helpers/Python/ContainerIO.h"
#include "revng/PipeboxCommon/Helpers/Python/Registry.h"
#include "revng/PipeboxCommon/Helpers/Python/RunAnalysis.h"
#include "revng/PipeboxCommon/Helpers/Python/RunPipe.h"
#include "revng/PipeboxCommon/Helpers/Python/SignatureHelper.h"

template<IsAnalysis T>
struct RegisterAnalysis {
  RegisterAnalysis() {
    using namespace nanobind::literals;
    using namespace revng::pypeline::helpers;

    // Python
    python::Registry.registerModuleInitializer([](nanobind::module_ &M,
                                                  python::BaseClasses &BC) {
      nanobind::class_<T>(M, T::Name.data(), BC.BaseAnalysis)
        .def_ro_static("name", &T::Name)
        .def(nanobind::init<>())
        .def_static("signature",
                    &python::SignatureHelper<T>::getSignature,
                    nanobind::sig("def signature() -> "
                                  "tuple[type[revng.pypeline.container."
                                  "Container], ...]"))
        .def("run",
             &python::runAnalysis<T>,
             "model"_a,
             "containers"_a,
             "incoming"_a,
             "configuration"_a);
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
        .def("objects", &python::ContainerIO<T>::objects)
        .def("verify", &T::verify)
        .def("deserialize", &python::ContainerIO<T>::deserialize)
        .def("serialize", &python::ContainerIO<T>::serialize);
    });
  }
};

template<IsPipe T>
struct RegisterPipe {
  RegisterPipe() {
    using namespace nanobind::literals;
    using namespace revng::pypeline::helpers;

    // Python
    python::Registry.registerModuleInitializer([](nanobind::module_ &M,
                                                  python::BaseClasses &BC) {
      nanobind::class_<T>(M, T::Name.data(), BC.BasePipe)
        .def_ro_static("name", &T::Name)
        .def_static("signature",
                    &python::SignatureHelper<T>::getSignature,
                    nanobind::sig("def signature() -> "
                                  "tuple[revng.pypeline.task.task.TaskArgument,"
                                  " ...]"))
        .def(nanobind::init<llvm::StringRef>())
        .def_prop_ro("static_configuration",
                     [](T &Handle) { return Handle.StaticConfiguration; })
        .def("run",
             &python::runPipe<T>,
             "model"_a,
             "containers"_a,
             "incoming"_a,
             "outgoing"_a,
             "configuration"_a);
    });
  }
};
