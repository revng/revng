#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/StringRef.h"

#include "revng/Pypeline/Common.h"
#include "revng/Pypeline/Concepts.h"
#include "revng/Pypeline/Container.h"
#include "revng/Pypeline/Model.h"
#include "revng/Pypeline/Registry.h"
#include "revng/Pypeline/StringRefCaster.h"
#include "revng/Pypeline/Utils.h"
#include "revng/Support/Assert.h"

#include "nanobind/stl/pair.h"
#include "nanobind/stl/string.h"
#include "nanobind/stl/vector.h"

namespace detail {

template<typename T>
class NanobindPipeRunner {
public:
  static detail::ObjectDependencies run(T *Handle,
                                        nanobind::handle_t<Model> TheModel,
                                        nanobind::list Containers,
                                        nanobind::list Incoming,
                                        nanobind::list Outgoing,
                                        nanobind::str Configuration) {
    const Model *CppModel = nanobind::cast<Model *>(TheModel);
    detail::RequestT CppIncoming = detail::convertRequests(Incoming);
    detail::RequestT CppOutgoing = detail::convertRequests(Outgoing);
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

template<IsPipe T>
class PipeWrapper final : public tracerunner::Pipe {
private:
  T Instance;

public:
  PipeWrapper(llvm::StringRef Conf) : Instance(Conf) {}
  ~PipeWrapper() override = default;

  virtual ObjectDependencies
  run(const Model *TheModel,
      std::vector<tracerunner::Container *> Containers,
      detail::RequestT Incoming,
      detail::RequestT Outgoing,
      llvm::StringRef Configuration) override {
    return PipeRunner<TraceRunnerInfo<T>>::runImpl(Instance,
                                                   &T::run,
                                                   TheModel,
                                                   Incoming,
                                                   Outgoing,
                                                   Configuration,
                                                   Containers);
  }
};

} // namespace detail

template<IsPipe T>
struct RegisterPipe {
  RegisterPipe() {
    TheRegistry.registerPythonCallback([](nanobind::module_ &M) {
      nanobind::object BaseClass = detail::getBaseClass("Pipe");
      nanobind::class_<T>(M, T::Name.data(), BaseClass)
        .def(nanobind::init<llvm::StringRef>())
        .def("run", &detail::NanobindPipeRunner<T>::run);
    });
    TheRegistry.registerTraceRunnerCallback([](TraceRunnerRegistry &R) {
      revng_assert(R.Pipes.count(T::Name) == 0);
      R.Pipes[T::Name] = [](llvm::StringRef Config) {
        return std::make_unique<detail::PipeWrapper<T>>(Config);
      };
    });
  }
};
