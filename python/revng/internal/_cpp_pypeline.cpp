//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "nanobind/nanobind.h"

#include "revng/Pypeline/Model.h"
#include "revng/Pypeline/Registrators/Detail/Casters.h"
#include "revng/Pypeline/Registrators/Detail/PythonUtils.h"
#include "revng/Pypeline/Registry.h"
#include "revng/Support/InitRevng.h"

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

  // Register the Buffer class
  nanobind::class_<pypeline::Buffer>(m, "Buffer")
    .def(nanobind::init<>())
    .def("__buffer__", [](pypeline::Buffer *Handle, int Flags) {
      return nanobind::steal(PyMemoryView_FromMemory(Handle->get().data(),
                                                     Handle->get().size(),
                                                     Flags));
    });

  // Register ObjectID
  nanobind::object ObjectIDBaseClass = detail::getBaseClass("ObjectID");
  nanobind::class_<ObjectID>(m, "ObjectID", ObjectIDBaseClass)
    .def(nanobind::init<>())
    .def("parent", &ObjectID::parent)
    .def("serialize", &ObjectID::serialize)
    .def("deserialize", &ObjectID::deserialize)
    .def("__eq__",
         [](ObjectID *Handle, nanobind::object Other) {
           if (not nanobind::isinstance<ObjectID>(Other))
             return false;
           ObjectID *OtherHandle = nanobind::cast<ObjectID *>(Other);
           return *Handle == *OtherHandle;
         })
    .def("__hash__",
         [](ObjectID *Handle) { return std::hash<ObjectID>{}(*Handle); });

  // Register Model
  nanobind::object ModelBaseClass = detail::getBaseClass("Model");
  nanobind::class_<Model>(m, "Model", ModelBaseClass)
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
    .def("serialize", &Model::serialize)
    .def("deserialize", &Model::deserialize);

  // Register all Pipes, Analyses and Containers
  pypeline::TheRegistry.callAll(m);
}
