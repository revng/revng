#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/StringRef.h"

#include "revng/Pypeline/Container.h"
#include "revng/Pypeline/Model.h"
#include "revng/Pypeline/Registry.h"
#include "revng/Pypeline/Utils.h"
#include "revng/Support/Assert.h"

template<typename T>
struct IsAnalysisRunMethod {
  static constexpr bool Value = false;
};

template<typename ReturnT, typename C, typename... ArgsT>
  requires(std::same_as<ReturnT, bool>
           && (IsContainer<std::remove_reference_t<ArgsT>> && ...))
struct IsAnalysisRunMethod<
  ReturnT (C::*)(Model *, detail::RequestT, llvm::StringRef, ArgsT...)> {
  static constexpr bool Value = true;
};

template<typename T>
concept IsAnalysis = requires(T A) {
  { T::Name } -> std::same_as<const llvm::StringRef &>;
} && IsAnalysisRunMethod<decltype(&T::run)>::Value;

template<typename T>
struct AnalysisRunner {
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

    return runImpl(*Handle,
                   &T::run,
                   CppModel,
                   CppIncoming,
                   CppConfiguration,
                   Containers);
  }

  template<typename... ContainersT>
    requires(IsContainer<std::remove_reference_t<ContainersT>> && ...)
  static bool runImpl(T &Analysis,
                      bool (T::*RunMethod)(Model *,
                                           detail::RequestT,
                                           llvm::StringRef,
                                           ContainersT...),
                      Model *TheModel,
                      detail::RequestT Incoming,
                      llvm::StringRef Configuration,
                      nanobind::list &Containers) {
    auto
      Sequence = std::make_integer_sequence<size_t, sizeof...(ContainersT)>();
    return ([&]<size_t... ContainerIndexes>(const integer_sequence<
                                            ContainerIndexes...> &) {
      return (Analysis.*RunMethod)(TheModel,
                                   Incoming,
                                   Configuration,
                                   detail::unwrapContainer<
                                     ContainersT,
                                     ContainerIndexes>(Containers)...);
    })(Sequence);
  }
};

template<IsAnalysis T>
struct RegisterAnalysis {
  RegisterAnalysis() {
    TheRegistry.registerPythonCallback([](nanobind::module_ &M) {
      nanobind::object BaseClass = detail::getBaseClass("Analysis");
      nanobind::class_<T>(M, T::Name.data(), BaseClass)
        .def_ro_static("_class_type", &"Analysis")
        .def(nanobind::init<>())
        .def("run", &AnalysisRunner<T>::run);
    });
  }
};
