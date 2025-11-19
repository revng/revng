//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <csignal>

#include "nanobind/nanobind.h"
#include "nanobind/stl/optional.h"
#include "nanobind/stl/set.h"
#include "nanobind/stl/string.h"
#include "nanobind/stl/vector.h"

#include "llvm/Support/Signals.h"

#include "revng/ADT/SetOperations.h"
#include "revng/PipeboxCommon/Helpers/Python/Casters.h"
#include "revng/PipeboxCommon/Helpers/Python/Helpers.h"
#include "revng/PipeboxCommon/Helpers/Python/Registry.h"
#include "revng/PipeboxCommon/Model.h"
#include "revng/Support/InitRevng.h"

struct SignalHandler {
  bool IsTerminating;
  sighandler_t Handler;
};

static std::map<int, SignalHandler> SavedSignals;

static void handleSignal(int SigNo) {
  const SignalHandler &Handler = SavedSignals[SigNo];
  if (Handler.IsTerminating)
    llvm::sys::RunInterruptHandlers();
  Handler.Handler(SigNo);
}

NB_MODULE(_pipebox, m) {
  using namespace revng::pypeline::helpers::python;

  auto Initialize = [m](std::set<int> TerminatingSignals,
                        std::set<int> NonterminatingSignals,
                        std::vector<std::string> ArgVector) {
    revng_assert(not nanobind::hasattr(m, "__init_revng__"));
    revng_assert(not intersects(TerminatingSignals, NonterminatingSignals));

    // Save the signal pointers for later
    for (int SigNumber :
         llvm::concat<const int>(TerminatingSignals, NonterminatingSignals)) {
      sighandler_t Handler = signal(SigNumber, SIG_DFL);
      if (Handler != SIG_ERR && Handler != NULL) {
        bool IsTerminating = TerminatingSignals.contains(SigNumber);
        SavedSignals[SigNumber] = { IsTerminating, Handler };
      }
    }

    int Argc = ArgVector.size() + 1;
    const char *Argv[Argc];
    Argv[0] = "";
    for (size_t I = 0; I < ArgVector.size(); I++)
      Argv[I + 1] = ArgVector[I].c_str();

    const char **ArgvPtr = Argv;
    // use a capsule to call the destructor when the Python module is unloaded
    m.attr("__init_revng__") = makeCapsule<revng::InitRevng>(Argc, ArgvPtr, "");

    for (int SigNumber : std::ranges::views::keys(SavedSignals))
      signal(SigNumber, &handleSignal);
  };
  m.def("initialize", Initialize);

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
    .def_static("model_name", []() { return nanobind::str("model.yml"); })
    .def_static("mime_type",
                []() { return nanobind::str("application/x-yaml"); })
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
