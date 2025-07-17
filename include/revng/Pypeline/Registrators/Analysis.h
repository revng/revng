#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/StringRef.h"

#include "revng/Pypeline/Concepts.h"
#include "revng/Pypeline/Model.h"
#include "revng/Pypeline/Registrators/Detail/PythonUtils.h"
#include "revng/Pypeline/Registry.h"
#include "revng/Pypeline/TraceRunner/Registry.h"
#include "revng/Support/Assert.h"

namespace detail {

template<typename Info>
struct AnalysisRunner {
private:
  template<size_t... I>
  using integer_sequence = std::integer_sequence<size_t, I...>;

  using T = Info::Type;
  using ListT = Info::ListT;

public:
  template<typename... ContainersT>
  static bool runImpl(T &Analysis,
                      bool (T::*RunMethod)(Model *,
                                           pypeline::RequestT,
                                           llvm::StringRef,
                                           ContainersT...),
                      Model *TheModel,
                      pypeline::RequestT Incoming,
                      llvm::StringRef Configuration,
                      ListT Containers) {
    revng_assert(Incoming.size() == sizeof...(ContainersT));
    auto
      Sequence = std::make_integer_sequence<size_t, sizeof...(ContainersT)>();
    return ([&]<size_t... ContainerIndexes>(const integer_sequence<
                                            ContainerIndexes...> &) {
      return (Analysis.*RunMethod)(TheModel,
                                   Incoming,
                                   Configuration,
                                   Info::template unwrap<
                                     ContainersT,
                                     ContainerIndexes>(Containers)...);
    })(Sequence);
  }
};

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
    pypeline::RequestT CppIncoming = detail::convertRequests(Incoming);
    llvm::StringRef CppConfiguration(Configuration.c_str());

    return AnalysisRunner<NanobindRunnerInfo<T>>::runImpl(*Handle,
                                                          &T::run,
                                                          CppModel,
                                                          CppIncoming,
                                                          CppConfiguration,
                                                          Containers);
  }
};

template<IsAnalysis T>
class AnalysisWrapper final : public pypeline::tracerunner::Analysis {
private:
  T Instance;

public:
  AnalysisWrapper() : Instance() {}
  ~AnalysisWrapper() override = default;

  virtual bool run(Model *TheModel,
                   std::vector<pypeline::tracerunner::Container *> Containers,
                   pypeline::RequestT Incoming,
                   llvm::StringRef Configuration) override {
    return AnalysisRunner<TraceRunnerInfo<T>>::runImpl(Instance,
                                                       &T::run,
                                                       TheModel,
                                                       Incoming,
                                                       Configuration,
                                                       Containers);
  }
};

} // namespace detail

template<IsAnalysis T>
struct RegisterAnalysis {
  RegisterAnalysis() {
    pypeline::TheRegistry.registerPythonCallback([](nanobind::module_ &M) {
      nanobind::object BaseClass = detail::getBaseClass("Analysis");
      nanobind::class_<T>(M, T::Name.data(), BaseClass)
        .def(nanobind::init<>())
        .def("run", &detail::NanobindAnalysisRunner<T>::run);
    });
    pypeline::TheRegistry
      .registerTraceRunnerCallback([](pypeline::tracerunner::Registry &R) {
        revng_assert(R.Analyses.count(T::Name) == 0);
        R.Analyses[T::Name] = []() {
          return std::make_unique<detail::AnalysisWrapper<T>>();
        };
      });
  }
};
