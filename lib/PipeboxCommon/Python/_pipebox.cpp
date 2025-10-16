//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "nanobind/nanobind.h"
#include "nanobind/stl/optional.h"
#include "nanobind/stl/set.h"
#include "nanobind/stl/string.h"
#include "nanobind/stl/vector.h"

#include "revng/PipeboxCommon/Helpers/Python/Casters.h"
#include "revng/PipeboxCommon/Helpers/Python/Helpers.h"
#include "revng/PipeboxCommon/Helpers/Python/Registry.h"
#include "revng/PipeboxCommon/Model.h"
#include "revng/Support/InitRevng.h"

NB_MODULE(_pipebox, m) {
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

  // Register the Buffer class
  nanobind::class_<revng::pypeline::Buffer>(m, "Buffer")
    .def(nanobind::init<>())
    .def("__buffer__", [](revng::pypeline::Buffer &Handle, int Flags) {
      // PyMemoryView_FromMemory returns a new reference, hence the need to
      // `steal` it with nanobind.
      return nanobind::steal(PyMemoryView_FromMemory(Handle.data().data(),
                                                     Handle.data().size(),
                                                     Flags));
    });

  // Register ObjectID
  nanobind::object ObjectIDBaseClass = importObject("revng.pypeline.object."
                                                    "ObjectID");
  nanobind::class_<ObjectID>(m, "ObjectID", ObjectIDBaseClass)
    .def(nanobind::init<>())
    .def("kind", &ObjectID::kind)
    .def("parent", &ObjectID::parent)
    .def_static("root", &ObjectID::root)
    .def("serialize", &ObjectID::serialize)
    .def_static("deserialize", &ObjectID::deserialize)
    .def("to_bytes",
         [](ObjectID &Handle) {
           std::vector<uint8_t> Result = Handle.toBytes();
           return nanobind::bytes(reinterpret_cast<void *>(Result.data()),
                                  Result.size());
         })
    .def_static("from_bytes",
                [](nanobind::bytes Bytes) {
                  const uint8_t
                    *Ptr = reinterpret_cast<const uint8_t *>(Bytes.data());
                  return ObjectID::fromBytes({ Ptr, Bytes.size() });
                })
    .def("__eq__",
         [](ObjectID &Handle, nanobind::object Other) {
           ObjectID *OtherHandle;
           if (not nanobind::try_cast<ObjectID *>(Other, OtherHandle))
             return false;
           return Handle == *OtherHandle;
         })
    .def("__hash__",
         [](ObjectID &Handle) { return std::hash<ObjectID>{}(Handle); });

  // Register Kind
  nanobind::object KindBaseClass = importObject("revng.pypeline.object.Kind");
  nanobind::class_<Kind>(m, "Kind", KindBaseClass)
    .def_static("kinds", &Kind::kinds)
    .def("parent", &Kind::parent)
    .def_static("deserialize", &Kind::deserialize)
    .def("serialize", &Kind::serlialize)
    .def("byte_size", &Kind::byteSize)
    .def("__eq__",
         [](Kind &Handle, nanobind::object Other) {
           Kind *OtherHandle;
           if (not nanobind::try_cast<Kind *>(Other, OtherHandle))
             return false;
           return Handle == *OtherHandle;
         })
    .def("__hash__", [](Kind &Handle) { return std::hash<Kind>{}(Handle); });

  // Register Model
  nanobind::object ModelBaseClass = importObject("revng.pypeline.model.Model");
  nanobind::class_<Model>(m, "Model", ModelBaseClass)
    .def(nanobind::init<>())
    .def("diff",
         [](Model &Handle, nanobind::handle_t<Model> Other) {
           return Handle.diff(*nanobind::cast<Model *>(Other));
         })
    .def("children", &Model::children)
    .def("clone", &Model::clone)
    .def("serialize",
         [](Model &Handle) {
           llvm::SmallVector<char, 0> Buffer = Handle.serialize();
           // TODO: this copies the data from the buffer, this cannot be
           //       avoided as Python does not have a way to "move" data into
           //       a bytes object. We could return `Buffer` but then a lot of
           //       libraries (e.g. `yaml`) would need to convert to bytes and
           //       copy anyways.
           return nanobind::bytes(Buffer.data(), Buffer.size());
         })
    .def_static("deserialize", &Model::deserialize);

  // Register all Pipes, Analyses and Containers
  BaseClasses BC{
    .BaseContainer = importObject("revng.pypeline.container.Container"),
    .BaseAnalysis = importObject("revng.pypeline.analysis.Analysis"),
    .BasePipe = importObject("revng.pypeline.task.pipe.Pipe"),
  };
  Registry.callAll(m, BC);
}
