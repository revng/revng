#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Pypeline/ObjectIDImpl.h"
#include "revng/Pypeline/Registry.h"
#include "revng/Pypeline/StringRefCaster.h"
#include "revng/Pypeline/Utils.h"

#include "nanobind/stl/optional.h"

struct RegisterObjectID {
  RegisterObjectID() {
    TheRegistry.registerPythonCallback([](nanobind::module_ &M) {
      nanobind::object BaseClass = detail::getBaseClass("ObjectID");
      nanobind::class_<ObjectID>(M, ObjectID::Name.data(), BaseClass)
        .def(nanobind::init<>())
        .def("components", &ObjectID::components)
        .def("parent", &ObjectID::parent)
        .def("serialize", &ObjectID::serialize)
        .def("deserialize", &ObjectID::deserialize)
        .def("__eq__",
             [](ObjectID *Handle, nanobind::object Other) {
               if (not nanobind::isinstance<ObjectID>(Other))
                 return false;
               ObjectID *OtherHandle = nanobind::cast<ObjectID *>(Other);
               return (*Handle <=> *OtherHandle) == 0;
             })
        .def("__hash__", [](ObjectID *Handle) {
          std::string Serialized(Handle->serialize());
          nanobind::str Str(Serialized.c_str());
          return nanobind::hash(Str);
        });
    });
  }
};
