#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/StringRef.h"

#include "revng/Pypeline/Common.h"
#include "revng/Pypeline/Container.h"
#include "revng/Pypeline/Model.h"
#include "revng/Pypeline/Registry.h"
#include "revng/Support/Assert.h"

#include "nanobind/stl/pair.h"
#include "nanobind/stl/string.h"
#include "nanobind/stl/vector.h"

template<typename T>
struct IsPipeRunMethod {
  static constexpr bool Value = false;
};

template<typename ReturnT, typename C, typename... ArgsT>
  requires(std::same_as<ReturnT, detail::ObjectDependencies>
           && (IsContainer<std::remove_reference_t<ArgsT>> && ...))
struct IsPipeRunMethod<ReturnT (C::*)(const Model *,
                                      detail::RequestT,
                                      detail::RequestT,
                                      llvm::StringRef,
                                      ArgsT...)> {
  static constexpr bool Value = true;
};

template<typename T>
concept IsPipe = requires(T A) {
  { T::Name } -> std::same_as<const llvm::StringRef &>;
} && IsPipeRunMethod<decltype(&T::run)>::Value;

template<typename T>
struct PipeRunner {
  template<size_t... I>
  using integer_sequence = std::integer_sequence<size_t, I...>;
  using ObjectDeps = detail::ObjectDependencies;

  static ObjectDeps run(T *Handle,
                        nanobind::handle_t<Model> TheModel,
                        nanobind::list Containers,
                        nanobind::list Incoming,
                        nanobind::list Outgoing,
                        nanobind::str Configuration) {
    const Model *CppModel = nanobind::cast<Model *>(TheModel);
    detail::RequestT CppIncoming = detail::convertRequests(Incoming);
    detail::RequestT CppOutgoing = detail::convertRequests(Outgoing);
    llvm::StringRef CppConfiguration(Configuration.c_str());

    return runImpl(*Handle,
                   &T::run,
                   CppModel,
                   CppIncoming,
                   CppOutgoing,
                   CppConfiguration,
                   Containers);
  }

  template<typename... ContainersT>
  static ObjectDeps runImpl(T &Pipe,
                            ObjectDeps (T::*RunMethod)(const Model *,
                                                       detail::RequestT,
                                                       detail::RequestT,
                                                       llvm::StringRef,
                                                       ContainersT...),
                            const Model *TheModel,
                            detail::RequestT Incoming,
                            detail::RequestT Outgoing,
                            llvm::StringRef Configuration,
                            nanobind::list &Containers) {
    revng_assert(Incoming.size() == sizeof...(ContainersT));
    revng_assert(Outgoing.size() == sizeof...(ContainersT));
    auto
      Sequence = std::make_integer_sequence<size_t, sizeof...(ContainersT)>();
    return ([&]<size_t... ContainerIndexes>(const integer_sequence<
                                            ContainerIndexes...> &) {
      return (Pipe.*RunMethod)(TheModel,
                               Incoming,
                               Outgoing,
                               Configuration,
                               detail::unwrapContainer<
                                 ContainersT,
                                 ContainerIndexes>(Containers)...);
    })(Sequence);
  }
};

template<IsPipe T>
struct RegisterPipe {
  RegisterPipe() {
    TheRegistry.registerPythonCallback([](nanobind::module_ &M) {
      nanobind::class_<T>(M, T::Name.data())
        .def(nanobind::init<>())
        .def("run", &PipeRunner<T>::run);
    });
  }
};
