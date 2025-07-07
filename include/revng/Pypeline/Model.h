#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Pypeline/ModelImpl.h"
#include "revng/Pypeline/Registry.h"
#include "revng/Pypeline/StringRefCaster.h"
#include "revng/Pypeline/Utils.h"

struct RegisterModel {
  RegisterModel() {
    TheRegistry.registerPythonCallback([](nanobind::module_ &M) {
      nanobind::object BaseClass = detail::getBaseClass("Model");
      nanobind::class_<Model>(M, Model::Name.data(), BaseClass)
        .def(nanobind::init<>())
        .def("diff",
             [](Model *Handle, nanobind::handle_t<Model> Other) {
               return Handle->diff(*nanobind::cast<Model *>(Other));
             })
        .def("clone",
             [](Model *Handle) {
               Model Cloned = Handle->clone();
               return nanobind::cast<Model>(std::move(Cloned));
             })
        .def("serialize",
             [](Model *Handle) {
               detail::Buffer Buffer = Handle->serialize();
               llvm::ArrayRef<char> Ref = Buffer.release();
               return nanobind::bytes(Ref.data(), Ref.size());
             })
        .def("deserialize", &Model::deserialize);
    });
  }
};
