#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <functional>
#include <map>
#include <set>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

#include "revng/Pypeline/Registry.h"
#include "revng/Pypeline/Utils.h"
#include "revng/Support/Assert.h"

#include "nanobind/stl/set.h"
#include "nanobind/stl/string.h"
#include "nanobind/stl/string_view.h"

template<typename T>
concept IsContainer = requires(T A,
                               const T AConst,
                               std::map<ObjectID *, llvm::ArrayRef<char>>
                                 InputData,
                               std::set<ObjectID *> OutputSet) {
  { T::Name } -> std::same_as<const llvm::StringRef &>;
  { T() };
  { AConst.kind() } -> std::same_as<ObjectID::Kind>;
  { AConst.objects() } -> std::same_as<std::set<ObjectID>>;
  { AConst.verify() } -> std::same_as<bool>;
  { A.deserialize(InputData) } -> std::same_as<void>;
  {
    AConst.serialize(OutputSet)
  } -> std::same_as<std::map<ObjectID, detail::Buffer>>;
};

template<typename T>
struct ContainerSerdes {
  static void deserialize(T *Handle, nanobind::dict &Data) {
    std::map<ObjectID *, llvm::ArrayRef<char>> Input;
    for (auto It = Data.begin(); It != Data.end(); It.increment()) {
      nanobind::bytes Second;
      ObjectID *First = nanobind::cast<ObjectID *>((*It).first);
      bool Result = nanobind::try_cast((*It).second, Second);
      revng_assert(Result);
      Input[First] = llvm::ArrayRef<
        char>(reinterpret_cast<const char *>(Second.data()), Second.size());
    }
    Handle->deserialize(Input);
  }

  static nanobind::dict serialize(T *Handle, nanobind::set Objects) {
    std::set<ObjectID *> CppObjects;
    for (auto It = Objects.begin(); It != Objects.end(); ++It) {
      revng_assert(nanobind::isinstance<ObjectID>(*It));
      CppObjects.insert(nanobind::cast<ObjectID *>(*It));
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

template<IsContainer T>
struct RegisterContainer {
  RegisterContainer() {
    TheRegistry.registerPythonCallback([](nanobind::module_ &M) {
      nanobind::object BaseClass = detail::getBaseClass("Container");
      nanobind::class_<T>(M, T::Name.data(), BaseClass)
        .def(nanobind::init<>())
        .def("kind", &T::kind)
        .def("objects", &T::objects)
        .def("verify", &T::verify)
        .def("deserialize", &ContainerSerdes<T>::deserialize)
        .def("serialize", &ContainerSerdes<T>::serialize);
    });
  }
};
