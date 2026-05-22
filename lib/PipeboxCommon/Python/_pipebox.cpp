//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <csignal>

#include "nanobind/nanobind.h"
#include "nanobind/stl/optional.h"
#include "nanobind/stl/pair.h"
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

static std::map<int, sighandler_t> SavedSignals;

static void handleSignal(int SigNo) {
  const sighandler_t &Handler = SavedSignals[SigNo];

  {
    // re-acquire the GIL since the signal might have been received while a pipe
    // is running. This calls `PyGILState_Ensure` which is re-entrant so it's
    // safe to call even if the GIL is currently held.
    nanobind::gil_scoped_acquire X;

    if (SigNo == SIGINT) {
      // At interpreter shutdown, nanobind checks if any of the objects it has
      // allocated are still alive and reports them as leaks. Since we're
      // calling `Py_Exit` the garbage collector is not run and so all
      // variables are still alive (but their destructors are still called).
      nanobind::set_leak_warnings(false);
      Py_Exit(1);
    } else {
      // Dispatch to the original python signal handler. If an exception is
      // raised its throwing might be delayed if there is a piece of native
      // code currently running.
      // Running the original python handler without currently holding the GIL
      // leads to a crash.
      Handler(SigNo);
    }
  }
}

static Logger ModelMigrationLogger("pypeline-model-migration");

/// This function serializes the model following the interface of
/// `Model.deserialize` in python. It tries to migrate old models if the first
/// time around the parsing fails. Returns the model object and a bool
/// indicating if a migration has happened.
static llvm::Expected<std::pair<Model, bool>>
deserializeModel(llvm::ArrayRef<uint8_t> Input) {
  using namespace revng::pypeline::helpers::python;

  auto MaybeModel = Model::deserialize(Input);
  if (MaybeModel)
    return std::pair{ *MaybeModel, false };

  revng_log(ModelMigrationLogger,
            "Model deserialization failed, attempting to migrate the model");
  nanobind::object MigrateFunction = importObject("revng.model.migrations."
                                                  "migrate_bytes");
  nanobind::object Output = MigrateFunction(nanobind::bytes(Input.data(),
                                                            Input.size()));
  // If the returned object is null there has been an exception, the input
  // model was probably broken, clear it out and propagate the first error.
  // TODO: wrap the python error into a `llvm::Error` and return the joined
  //       error
  if (Output.ptr() == nullptr) {
    revng_log(ModelMigrationLogger, "Model migration failed, bailing out");
    revng_assert(PyErr_Occurred() != NULL);
    PyErr_Clear();
    return MaybeModel.takeError();
  }

  revng_log(ModelMigrationLogger,
            "Model migration succeedeed, attempting to re-deserialize the "
            "model");
  auto OutputBytes = nanobind::cast<nanobind::bytes>(Output);
  auto *BytesPointer = static_cast<const uint8_t *>(OutputBytes.data());
  auto MaybeModel2 = Model::deserialize({ BytesPointer, OutputBytes.size() });
  if (MaybeModel2) {
    // Here consumeError is valid because the error came from the fact that the
    // model was of a previous version and it got updated
    llvm::consumeError(MaybeModel.takeError());
    return std::pair{ *MaybeModel2, true };
  }

  revng_log(ModelMigrationLogger, "Model re-deserialization failed, bailing");
  // If here, we didn't manage to deserialize a valid model, return the errors
  return revng::createError("Invalid model was migrated but remains invalid\n"
                            + llvm::toString(MaybeModel.takeError()) + "\n"
                            + llvm::toString(MaybeModel2.takeError()));
}

NB_MODULE(_pipebox, m) {
  using namespace revng::pypeline::helpers::python;

  auto Initialize = [m](std::set<int> Signals,
                        std::vector<std::string> ArgVector) {
    revng_assert(not nanobind::hasattr(m, "__init_revng__"));

    // Save the signal pointers for later
    for (int SigNumber : Signals) {
      sighandler_t Handler = signal(SigNumber, SIG_DFL);
      if (Handler != SIG_ERR && Handler != NULL)
        SavedSignals[SigNumber] = Handler;
    }

    // Register cleanup function for llvm
    int RC = Py_AtExit(&llvm::sys::RunInterruptHandlers);
    revng_assert(RC == 0);

    // The `signal` module saves internally the python functions stored via
    // `signal.signal`, this is required for `signal.getsignal` to work. Some
    // library code, e.g. `asyncio` behaves differently if the signal handler
    // (retrieved by signal) is the default one. To avoid inconsistencies we
    // manually change the signal handler (via `signal.signal`) to a dummy
    // function so that the special behavior is not triggered.
    nanobind::object SignalFunction = importObject("signal.signal");
    nanobind::object
      SignalHandler = nanobind::cpp_function([](int, nanobind::object) {
        return;
      });
    for (int SigNumber : std::ranges::views::keys(SavedSignals))
      SignalFunction(nanobind::int_(SigNumber), SignalHandler);

    // The arguments need to have the same storage duration as the InitRevng,
    // since they might be used for the crash handler. Store them in `static`
    // variables to avoid any problems.
    static std::vector<std::string> Args = { "" };
    append(ArgVector, Args);

    static std::vector<char *> Argv;
    for (std::string &String : Args)
      Argv.push_back(&String[0]);

    static int Argc = Argv.size();
    static char **ArgvPtr = Argv.data();
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

  // Register ModelDiff
  nanobind::object ModelDiffBaseClass = importObject("revng.pypeline.model."
                                                     "ModelDiff");
  nanobind::class_<ModelDiff>(m, "ModelDiff", ModelDiffBaseClass)
    .def("paths", &ModelDiff::paths)
    .def("serialize", [](ModelDiff &Handle) {
      llvm::SmallVector<char, 0> Buffer = Handle.serialize();
      return nanobind::bytes(Buffer.data(), Buffer.size());
    });

  // Register Model
  nanobind::object ModelBaseClass = importObject("revng.pypeline.model.Model");
  nanobind::class_<Model>(m, "Model", ModelBaseClass)
    .def(nanobind::init<>())
    .def_ro_static("identifier", &"revng")
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
    .def_static("deserialize", &deserializeModel)
    .def("__eq__",
         [](Model &Handle, nanobind::object Other) {
           Model *OtherHandle;
           if (not nanobind::try_cast<Model *>(Other, OtherHandle))
             return false;
           return Handle == *OtherHandle;
         })
    .def("enable_caching", &Model::enableCaching)
    .def("disable_caching", &Model::disableCaching);

  // Register all Pipes, Analyses and Containers
  BaseClasses BC{
    .BaseContainer = importObject("revng.pypeline.container.Container"),
    .BaseAnalysis = importObject("revng.pypeline.analysis.Analysis"),
    .BasePipe = importObject("revng.pypeline.task.pipe.Pipe"),
  };
  Registry.callAll(m, BC);
}
