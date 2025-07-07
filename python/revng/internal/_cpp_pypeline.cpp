//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "nanobind/nanobind.h"
#include "nanobind/stl/optional.h"
#include "nanobind/stl/set.h"
#include "nanobind/stl/string.h"

#include "revng/Pypeline/Helpers/Python/Casters.h"
#include "revng/Pypeline/Helpers/Python/Registry.h"
#include "revng/Pypeline/Model.h"
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
    nanobind::setattr(m, "__init_revng__", Capsule);
  }

  using namespace revng::pypeline::helpers::python;
  nanobind::module_ BaseModule = nanobind::module_::import_("revng.pypeline");

  // Register the Buffer class
  nanobind::class_<revng::pypeline::Buffer>(m, "Buffer")
    .def(nanobind::init<>())
    .def("__buffer__", [](revng::pypeline::Buffer *Handle, int Flags) {
      return nanobind::steal(PyMemoryView_FromMemory(Handle->get().data(),
                                                     Handle->get().size(),
                                                     Flags));
    });

  // Register ObjectID
  nanobind::object ObjectIDBaseClass = BaseModule.attr("ObjectID");
  nanobind::type_object ObjectIDRelation = ObjectIDBaseClass.attr("Relation");
  nanobind::class_<ObjectID> ObjectIDClass(m, "ObjectID", ObjectIDBaseClass);
  ObjectIDClass.def(nanobind::init<>())
    .def("kind", &ObjectID::kind)
    .def("parent", &ObjectID::parent)
    .def("serialize", &ObjectID::serialize)
    .def("deserialize", &ObjectID::deserialize)
    .def_static("kind_relation",
                [ObjectIDRelation](ObjectID::Kind From, ObjectID::Kind To) {
                  ObjectID::Relation Result = ObjectID::kindRelation(From, To);
                  using IntType = std::underlying_type_t<ObjectID::Relation>;
                  return ObjectIDRelation(static_cast<IntType>(Result));
                })
    .def("__eq__",
         [](ObjectID *Handle, nanobind::object Other) {
           if (not nanobind::isinstance<ObjectID>(Other))
             return false;
           ObjectID *OtherHandle = nanobind::cast<ObjectID *>(Other);
           return *Handle == *OtherHandle;
         })
    .def("__hash__",
         [](ObjectID *Handle) { return std::hash<ObjectID>{}(*Handle); });

  // Register ObjectID::Kind
  nanobind::enum_<ObjectID::Kind>(ObjectIDClass, "Kind")
    .value("Root", ObjectID::Kind::Root)
    .value("Function", ObjectID::Kind::Function)
    .value("TypeDefinition", ObjectID::Kind::TypeDefinition);

  // Register Model
  nanobind::object ModelBaseClass = BaseModule.attr("Model");
  nanobind::class_<Model>(m, "Model", ModelBaseClass)
    .def(nanobind::init<>())
    .def("diff",
         [](Model *Handle, nanobind::handle_t<Model> Other) {
           return Handle->diff(*nanobind::cast<Model *>(Other));
         })
    .def("clone", &Model::clone)
    .def("serialize", &Model::serialize)
    .def("deserialize", &Model::deserialize);

  // Register all Pipes, Analyses and Containers
  BaseClasses BC{
    .BaseContainer = BaseModule.attr("Container"),
    .BaseAnalysis = BaseModule.attr("Analysis"),
    .BasePipe = BaseModule.attr("Pipe"),
  };
  Registry.callAll(m, BC);
}
