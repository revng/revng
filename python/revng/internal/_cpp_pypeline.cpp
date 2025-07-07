//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Pypeline/Model.h"
#include "revng/Pypeline/Registrators/Detail/PythonUtils.h"
#include "revng/Pypeline/Registrators/Detail/StringRefCaster.h"
#include "revng/Pypeline/Registry.h"
#include "revng/Support/InitRevng.h"

#include "nanobind/nanobind.h"

NB_MODULE(_cpp_pypeline, m) {
  {
    // Do InitRevng, use a capsule to call the destructor when the Python module
    // is unloaded
    int Argc = 1;
    const char *Argv[] = { "" };
    const char **ArgvPtr = Argv;
    nanobind::capsule Capsule(new revng::InitRevng(Argc, ArgvPtr, "", {}),
                              [](void *Ptr) noexcept {
                                delete static_cast<revng::InitRevng *>(Ptr);
                              });
    nanobind::setattr(m, "__init_revng", Capsule);
  }

  // Register ObjectID
  nanobind::object ObjectIDBaseClass = detail::getBaseClass("ObjectID");
  nanobind::class_<ObjectID>(m, ObjectID::Name.data(), ObjectIDBaseClass)
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
    .def("__hash__", &ObjectID::hash);

  // Register Model
  nanobind::object ModelBaseClass = detail::getBaseClass("Model");
  nanobind::class_<Model>(m, Model::Name.data(), ModelBaseClass)
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
           pypeline::Buffer Buffer = Handle->serialize();
           llvm::ArrayRef<char> Ref = Buffer.release();
           return nanobind::bytes(Ref.data(), Ref.size());
         })
    .def("deserialize", &Model::deserialize);

  // Register all Pipes, Analyses and Containers
  pypeline::TheRegistry.callAll(m);
}
