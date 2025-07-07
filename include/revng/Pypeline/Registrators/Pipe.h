#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/StringRef.h"

#include "revng/Pypeline/Common.h"
#include "revng/Pypeline/Concepts.h"
#include "revng/Pypeline/Model.h"
#include "revng/Pypeline/Registrators/Detail/PythonUtils.h"
#include "revng/Pypeline/Registrators/Detail/StringRefCaster.h"
#include "revng/Pypeline/Registry.h"
#include "revng/Support/Assert.h"

#include "nanobind/stl/pair.h"
#include "nanobind/stl/string.h"
#include "nanobind/stl/vector.h"

namespace detail {

template<typename Info>
struct PipeRunner {
private:
  template<size_t... I>
  using integer_sequence = std::integer_sequence<size_t, I...>;
  using ObjectDeps = pypeline::ObjectDependencies;

  using T = Info::Type;
  using ListT = Info::ListT;

public:
  template<typename... ContainersT>
  static ObjectDeps runImpl(T &Pipe,
                            ObjectDeps (T::*RunMethod)(const Model *,
                                                       pypeline::RequestT,
                                                       pypeline::RequestT,
                                                       llvm::StringRef,
                                                       ContainersT...),
                            const Model *TheModel,
                            pypeline::RequestT Incoming,
                            pypeline::RequestT Outgoing,
                            llvm::StringRef Configuration,
                            ListT Containers) {
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
                               Info::template unwrap<
                                 ContainersT,
                                 ContainerIndexes>(Containers)...);
    })(Sequence);
  }
};

template<typename T>
class NanobindPipeRunner {
public:
  static pypeline::ObjectDependencies run(T *Handle,
                                          nanobind::handle_t<Model> TheModel,
                                          nanobind::list Containers,
                                          nanobind::list Incoming,
                                          nanobind::list Outgoing,
                                          nanobind::str Configuration) {
    const Model *CppModel = nanobind::cast<Model *>(TheModel);
    pypeline::RequestT CppIncoming = detail::convertRequests(Incoming);
    pypeline::RequestT CppOutgoing = detail::convertRequests(Outgoing);
    llvm::StringRef CppConfiguration(Configuration.c_str());

    return PipeRunner<NanobindRunnerInfo<T>>::runImpl(*Handle,
                                                      &T::run,
                                                      CppModel,
                                                      CppIncoming,
                                                      CppOutgoing,
                                                      CppConfiguration,
                                                      Containers);
  }
};

} // namespace detail

template<IsPipe T>
struct RegisterPipe {
  RegisterPipe() {
    pypeline::TheRegistry.registerPythonCallback([](nanobind::module_ &M) {
      nanobind::object BaseClass = detail::getBaseClass("Pipe");
      nanobind::class_<T>(M, T::Name.data(), BaseClass)
        .def(nanobind::init<llvm::StringRef>())
        .def("run", &detail::NanobindPipeRunner<T>::run);
    });
  }
};
