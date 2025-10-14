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
#include "revng/PipeboxCommon/Helpers/Native/Registry.h"
#include "revng/PipeboxCommon/Helpers/PipeRunPipes/FunctionPipe.h"
#include "revng/PipeboxCommon/Helpers/PipeRunPipes/LLVMFunctionPassPipe.h"
#include "revng/PipeboxCommon/Helpers/PipeRunPipes/SingleOutputPipe.h"
#include "revng/PipeboxCommon/Helpers/PipeRunPipes/TypeDefinitionPipe.h"
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
      std::string Name = T::Name.str();
      nanobind::class_<T>(M, Name.c_str(), BC.BaseAnalysis)
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
      std::string Name = T::Name.str();
      nanobind::class_<T>(M, Name.c_str(), BC.BaseContainer)
        .def_ro_static("kind", &T::Kind)
        .def_static("mime_type", []() { return T::MimeType; })
        .def(nanobind::init<>())
        .def("objects", &python::ContainerIO<T>::objects)
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
    using namespace nanobind::literals;
    using namespace revng::pypeline::helpers;

    // Python
    python::Registry.registerModuleInitializer([](nanobind::module_ &M,
                                                  python::BaseClasses &BC) {
      std::string Name = T::Name.str();
      auto PipeClass = nanobind::class_<T>(M, Name.c_str(), BC.BasePipe);
      PipeClass.def_ro_static("name", &T::Name)
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
             "file_provider"_a,
             "model"_a,
             "containers"_a,
             "incoming"_a,
             "outgoing"_a,
             "configuration"_a);

      if constexpr (HasCheckPrecondition<T>) {
        PipeClass
          .def("check_precondition", [](T &Handle, nanobind::object TheModel) {
            const Model &CppModel = python::convertReadOnlyModel(TheModel);
            return Handle.checkPrecondition(CppModel);
          });
      }
    });

    // Native
    revng_assert(native::Registry.Pipes.count(T::Name) == 0);
    native::Registry.Pipes[T::Name] =
      [](llvm::StringRef Config) -> std::unique_ptr<native::Pipe> {
      return std::make_unique<native::PipeImpl<T>>(Config);
    };
  }
};

template<typename T>
using RegisterSingleOutputPipe = RegisterPipe<SingleOutputPipe<T>>;

template<typename T>
using RegisterFunctionPipe = RegisterPipe<FunctionPipe<T>>;

template<typename T>
using RegisterTypeDefinitionPipe = RegisterPipe<TypeDefinitionPipe<T>>;

template<typename T>
using RegisterLLVMFunctionPassPipe = RegisterPipe<LLVMFunctionPassPipe<T>>;
