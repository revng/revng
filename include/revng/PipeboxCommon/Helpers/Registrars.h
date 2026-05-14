#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "nanobind/nanobind.h"
#include "nanobind/stl/optional.h"
#include "nanobind/stl/pair.h"
#include "nanobind/stl/set.h"
#include "nanobind/stl/string.h"
#include "nanobind/stl/tuple.h"
#include "nanobind/stl/vector.h"

#include "revng/PipeboxCommon/Concepts.h"
#include "revng/PipeboxCommon/Helpers/Native/Registry.h"
#include "revng/PipeboxCommon/Helpers/PipeRunPipes/FunctionPipe.h"
#include "revng/PipeboxCommon/Helpers/PipeRunPipes/SingleOutputPipe.h"
#include "revng/PipeboxCommon/Helpers/PipeRunPipes/TypeDefinitionPipe.h"
#include "revng/PipeboxCommon/Helpers/Python/Casters.h"
#include "revng/PipeboxCommon/Helpers/Python/ContainerIO.h"
#include "revng/PipeboxCommon/Helpers/Python/Invalidate.h"
#include "revng/PipeboxCommon/Helpers/Python/Registry.h"
#include "revng/PipeboxCommon/Helpers/Python/RunAnalysis.h"
#include "revng/PipeboxCommon/Helpers/Python/RunPipe.h"
#include "revng/PipeboxCommon/Helpers/Python/SignatureHelper.h"

inline Logger PypelineRegisterLogger("pypeline-register");

inline std::string pascalCaseName(llvm::StringRef Name) {
  std::string Result;
  bool ShouldCapitalizeNextCharacter = true;
  for (size_t I = 0; I < Name.size(); I++) {
    if (Name[I] == '-') {
      ShouldCapitalizeNextCharacter = true;
      continue;
    }

    if (ShouldCapitalizeNextCharacter) {
      Result.push_back(llvm::toUpper(Name[I]));
      ShouldCapitalizeNextCharacter = false;
    } else {
      Result.push_back(Name[I]);
    }
  }
  return Result;
}

template<IsAnalysis T>
struct RegisterAnalysis {
  RegisterAnalysis() {
    using namespace nanobind::literals;
    using namespace revng::pypeline::helpers;

    // Python
    python::Registry.registerModuleInitializer([](nanobind::module_ &M,
                                                  python::BaseClasses &BC) {
      std::string Name = pascalCaseName(T::Name);
      auto
        AnalysisClass = nanobind::class_<T>(M, Name.c_str(), BC.BaseAnalysis)
                          .def_ro_static("name", &T::Name)
                          .def(nanobind::init<>())
                          .def_static("signature",
                                      &python::SignatureHelper<T>::getSignature,
                                      nanobind::sig("def signature() -> "
                                                    "tuple[type[revng.pypeline."
                                                    "container."
                                                    "Container], ...]"))
                          .def("run",
                               &python::runAnalysis<T>,
                               "model"_a,
                               "containers"_a,
                               "incoming"_a,
                               "configuration"_a);

      if constexpr (HasIsAvailable<T>)
        AnalysisClass.def("is_available", &T::isAvailable);
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
      std::string Name = pascalCaseName(T::Name);
      nanobind::class_<T>(M, Name.c_str(), BC.BaseContainer)
        .def_ro_static("name", &T::Name)
        .def_ro_static("kind", &T::Kind)
        .def_ro_static("compression", &T::Compression)
        .def_static("mime_type", []() { return T::MimeType; })
        .def(nanobind::init<>())
        .def("objects", &python::ContainerIO<T>::objects)
        .def("verify", &T::verify)
        .def("set_is_disposable", &T::setIsDisposable)
        .def("dispose_if_possible", &T::disposeIfPossible)
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

namespace detail {

template<typename T>
void checkPipeArgumentAccess() {
  if (not PypelineRegisterLogger.isEnabled())
    return;

  using CT = PipeRunTraits<T>::ContainerTypes;
  using AccessEnum = revng::pypeline::Access;
  forEach<CT>([]<typename A, size_t I>() {
    using Argument = std::tuple_element_t<I, typename T::Arguments>;
    if (Argument::Access == AccessEnum::Read and not std::is_const_v<A>) {
      revng_log(PypelineRegisterLogger,
                T::Name << " has the " << I
                        << "th argument with READ access but marked non-const");
    }
  });
}

} // namespace detail

template<IsPipe T>
struct RegisterPipe {
  RegisterPipe() {
    using namespace nanobind::literals;
    using namespace revng::pypeline::helpers;

    detail::checkPipeArgumentAccess<T>();

    // Python
    python::Registry.registerModuleInitializer([](nanobind::module_ &M,
                                                  python::BaseClasses &BC) {
      std::string Name = pascalCaseName(T::Name);
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

      if constexpr (HasCustomInvalidation<T>) {
        PipeClass.def("requires_custom_invalidation",
                      [](T &Handle, nanobind::handle_t<ModelDiff> Diff) {
                        ModelDiff *CppDiff = nanobind::cast<ModelDiff *>(Diff);
                        return Handle.requiresCustomInvalidation(*CppDiff);
                      });
        PipeClass.def("process_custom_invalidation",
                      &python::processCustomInvalidation<T>);
      }
    });

    // Native
    if (native::Registry.Pipes.count(T::Name) != 0) {
      std::string Error = "Duplicate pipes: '" + T::Name.str() + "'";
      revng_abort(Error.c_str());
    }
    native::Registry.Pipes[T::Name] =
      [](llvm::StringRef Config) -> std::unique_ptr<native::Pipe> {
      return std::make_unique<native::PipeImpl<T>>(Config);
    };
  }
};

template<IsSingleOutputPipeRun T>
using RegisterSingleOutputPipeRun = RegisterPipe<SingleOutputPipe<T>>;

template<IsFunctionPipeRun T>
using RegisterFunctionPipeRun = RegisterPipe<FunctionPipe<T>>;

template<IsTypeDefinitionPipeRun T>
using RegisterTypeDefinitionPipeRun = RegisterPipe<TypeDefinitionPipe<T>>;
