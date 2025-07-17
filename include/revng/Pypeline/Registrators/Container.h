#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <functional>
#include <map>
#include <set>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

#include "revng/Pypeline/Concepts.h"
#include "revng/Pypeline/Registrators/Detail/PythonUtils.h"
#include "revng/Pypeline/Registry.h"
#include "revng/Pypeline/TraceRunner/Registry.h"
#include "revng/Support/Assert.h"

#include "nanobind/stl/set.h"
#include "nanobind/stl/string.h"
#include "nanobind/stl/string_view.h"

namespace detail {

template<typename T>
struct ContainerSerdes {
  static void deserialize(T *Handle, nanobind::dict &Data) {
    std::map<const ObjectID *, llvm::ArrayRef<const char>> Input;
    for (auto It = Data.begin(); It != Data.end(); It.increment()) {
      nanobind::bytes Second;
      ObjectID *First = nanobind::cast<ObjectID *>((*It).first);
      bool Result = nanobind::try_cast((*It).second, Second);
      revng_assert(Result);
      const char *DataPtr = reinterpret_cast<const char *>(Second.data());
      Input[First] = llvm::ArrayRef<const char>(DataPtr, Second.size());
    }
    Handle->deserialize(Input);
  }

  static nanobind::dict serialize(T *Handle, nanobind::set Objects) {
    std::vector<const ObjectID *> CppObjects;
    for (auto It = Objects.begin(); It != Objects.end(); ++It) {
      revng_assert(nanobind::isinstance<ObjectID>(*It));
      CppObjects.push_back(nanobind::cast<ObjectID *>(*It));
    }

    auto Result = Handle->serialize(CppObjects);
    nanobind::dict Return;
    for (auto &Entry : Result) {
      ObjectID KeyCopy = Entry.first;
      nanobind::object Key = nanobind::cast<ObjectID>(std::move(KeyCopy));
      llvm::ArrayRef<char> EntryData = Entry.second.release();
      nanobind::bytes Value(EntryData.data(), EntryData.size());
      Return[Key] = Value;
    }
    return Return;
  };
};

template<typename T>
class ContainerWrapper final : public pypeline::tracerunner::Container {
private:
  T Instance;

public:
  ContainerWrapper() : Instance() {}
  ~ContainerWrapper() override = default;

public:
  virtual void
  deserialize(const std::map<const ObjectID *, llvm::ArrayRef<const char>>
                &Input) override {
    Instance.deserialize(Input);
  }

  virtual std::map<ObjectID, pypeline::Buffer>
  serialize(llvm::ArrayRef<const ObjectID> ToSave) const override {
    std::vector<const ObjectID *> Input;
    for (const ObjectID &Element : ToSave)
      Input.push_back(&Element);

    return Instance.serialize(Input);
  }

public:
  virtual void *get() override { return &Instance; }
};

} // namespace detail

template<IsContainer T>
struct RegisterContainer {
  RegisterContainer() {
    pypeline::TheRegistry.registerPythonCallback([](nanobind::module_ &M) {
      nanobind::object BaseClass = detail::getBaseClass("Container");
      nanobind::class_<T>(M, T::Name.data(), BaseClass)
        .def(nanobind::init<>())
        .def("kind", &T::kind)
        .def("objects", &T::objects)
        .def("verify", &T::verify)
        .def("deserialize", &detail::ContainerSerdes<T>::deserialize)
        .def("serialize", &detail::ContainerSerdes<T>::serialize);
    });
    pypeline::TheRegistry
      .registerTraceRunnerCallback([](pypeline::tracerunner::Registry &R) {
        revng_assert(R.Containers.count(T::Name) == 0);
        R.Containers[T::Name] = []() {
          return std::make_unique<detail::ContainerWrapper<T>>();
        };
      });
  }
};
