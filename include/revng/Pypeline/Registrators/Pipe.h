#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "nanobind/stl/pair.h"
#include "nanobind/stl/string.h"
#include "nanobind/stl/vector.h"

#include "llvm/ADT/StringRef.h"

#include "revng/Pypeline/Common.h"
#include "revng/Pypeline/Concepts.h"
#include "revng/Pypeline/Model.h"
#include "revng/Pypeline/Registrators/Detail/Casters.h"
#include "revng/Pypeline/Registrators/Detail/PythonUtils.h"
#include "revng/Pypeline/Registry.h"
#include "revng/Support/Assert.h"

namespace detail {

/// Helper class that allows running a C++ analysis
/// The main hiccup in doing so is to unpack the arguments which are in some
/// kind of sequence (e.g. std::vector) into arguments for the Pipe::run
/// method. The type of the Pipe, the type of the sequence and how to unpack
/// them are conveyed through the Info type, which cannot be a function pointer
/// due to the unpacking function requiring template parameters.
template<typename Info>
struct PipeRunner {
private:
  template<size_t... I>
  using integer_sequence = std::integer_sequence<size_t, I...>;
  using ObjectDeps = pypeline::ObjectDependencies;

  using T = Info::Type;
  using ListType = Info::ListType;

public:
  template<typename... ContainersT>
  static ObjectDeps runImpl(T &Pipe,
                            ObjectDeps (T::*RunMethod)(const Model *,
                                                       pypeline::Request,
                                                       pypeline::Request,
                                                       llvm::StringRef,
                                                       ContainersT...),
                            const Model *TheModel,
                            pypeline::Request Incoming,
                            pypeline::Request Outgoing,
                            llvm::StringRef Configuration,
                            ListType Containers) {
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
class PythonPipeRunner {
public:
  static pypeline::ObjectDependencies run(T *Handle,
                                          nanobind::handle_t<Model> TheModel,
                                          nanobind::list Containers,
                                          nanobind::list Incoming,
                                          nanobind::list Outgoing,
                                          nanobind::str Configuration) {
    const Model *CppModel = nanobind::cast<Model *>(TheModel);
    pypeline::Request CppIncoming = detail::convertRequests(Incoming);
    pypeline::Request CppOutgoing = detail::convertRequests(Outgoing);
    llvm::StringRef CppConfiguration(Configuration.c_str());

    return PipeRunner<PythonRunnerInfo<T>>::runImpl(*Handle,
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
    pypeline::TheRegistry
      .registerPythonModuleInitializer([](nanobind::module_ &M) {
        nanobind::object BaseClass = detail::getBaseClass("Pipe");
        nanobind::class_<T>(M, T::Name.data(), BaseClass)
          .def(nanobind::init<llvm::StringRef>())
          .def("run", &detail::PythonPipeRunner<T>::run);
      });
  }
};
