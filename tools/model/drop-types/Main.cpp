//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <string>

#include "llvm/Support/CommandLine.h"

#include "revng/Model/Binary.h"
#include "revng/Model/Processing.h"
#include "revng/Support/Debug.h"
#include "revng/Support/InitRevng.h"
#include "revng/Support/YAMLTraits.h"

namespace cl = llvm::cl;

static cl::OptionCategory ThisToolCategory("model-drop-types options", "");

static cl::opt<std::string> InputFilename(cl::Positional,
                                          cl::cat(ThisToolCategory),
                                          cl::desc("<input model>"),
                                          cl::init("-"),
                                          cl::value_desc("model"));

static cl::opt<std::string> OutputFilename("o",
                                           cl::cat(ThisToolCategory),
                                           cl::init("-"),
                                           cl::desc("Override output "
                                                    "filename"),
                                           cl::value_desc("filename"));

static cl::list<std::string> TypeKeys(cl::Positional,
                                      cl::cat(ThisToolCategory),
                                      cl::desc("<type keys to drop...>"),
                                      cl::OneOrMore);

int main(int Argc, char *Argv[]) {
  revng::InitRevng X(Argc, Argv, "", { &ThisToolCategory });

  llvm::ExitOnError ExitOnError;

  auto MaybeModel = TupleTree<model::Binary>::fromFileOrSTDIN(InputFilename);
  if (not MaybeModel)
    ExitOnError(MaybeModel.takeError());

  auto &Model = *MaybeModel;

  // Resolve the type keys into TypeDefinition pointers
  std::set<const model::TypeDefinition *> TypesToDrop;
  for (const std::string &KeyString : TypeKeys) {
    // Keys have the form "ID-Kind", e.g. "42-StructDefinition"
    auto SplitPoint = KeyString.find('-');
    if (SplitPoint == std::string::npos) {
      dbg << "Invalid type key (expected ID-Kind): " << KeyString << "\n";
      return EXIT_FAILURE;
    }

    uint64_t ID = 0;
    if (llvm::StringRef(KeyString).substr(0, SplitPoint).getAsInteger(10, ID)) {
      dbg << "Invalid ID in type key: " << KeyString << "\n";
      return EXIT_FAILURE;
    }

    llvm::StringRef KindName = llvm::StringRef(KeyString).substr(SplitPoint
                                                                 + 1);
    auto Kind = model::TypeDefinitionKind::fromName(KindName);
    if (Kind == model::TypeDefinitionKind::Invalid) {
      dbg << "Unknown type kind: " << KindName.str() << "\n";
      return EXIT_FAILURE;
    }

    // Look up the type in the model
    bool Found = false;
    for (const model::UpcastableTypeDefinition &Type :
         Model->TypeDefinitions()) {
      if (Type->ID() == ID && Type->Kind() == Kind) {
        TypesToDrop.insert(Type.get());
        Found = true;
        break;
      }
    }

    if (not Found) {
      dbg << "Type not found in model: " << KeyString << "\n";
      return EXIT_FAILURE;
    }
  }

  unsigned Dropped = model::dropTypesDependingOnDefinitions(Model, TypesToDrop);

  dbg << "Dropped " << Dropped << " type(s)\n";

  ExitOnError(Model.toFile(OutputFilename));

  return EXIT_SUCCESS;
}
