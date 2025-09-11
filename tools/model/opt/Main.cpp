/// \file Main.cpp

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <string>

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"

#include "revng/Model/Pass/RegisterModelPass.h"
#include "revng/Model/Processing.h"
#include "revng/Support/InitRevng.h"

namespace cl = llvm::cl;
static cl::OptionCategory ThisToolCategory("Tool options", "");
extern cl::OptionCategory ModelPassCategory;

static cl::opt<std::string> OutputFilename("o",
                                           llvm::cl::init("-"),
                                           llvm::cl::desc("Override output "
                                                          "filename"),
                                           llvm::cl::value_desc("filename"),
                                           llvm::cl::cat(ThisToolCategory));

static cl::opt<std::string> InputFilename(cl::Positional,
                                          cl::desc("<input model file>"),
                                          cl::init("-"),
                                          cl::value_desc("filename"),
                                          cl::cat(ThisToolCategory));

class PassName : public std::string {
public:
  PassName() {}
  PassName(const std::string &String) : std::string(String) {}
};

namespace llvm {
namespace cl {
/// Define a valid OptionValue for the command line pass argument.
template<>
struct OptionValue<PassName> final
  : OptionValueBase<PassName, /*isClass=*/true> {
  OptionValue(const PassName &Value) { this->setValue(Value); }
  OptionValue() = default;
  void anchor() override {}

  bool hasValue() const { return true; }
  const PassName &getValue() const { return Value; }
  void setValue(const PassName &Value) { this->Value = Value; }

  PassName Value;
};

} // namespace cl
} // namespace llvm

static cl::list<PassName> PassesList(cl::desc("Optimizations available:"),
                                     cl::cat(ThisToolCategory));

static void loadPassesList() {
  for (const auto &[Name, Description, _] : RegisterModelPass::values())
    PassesList.getParser().addLiteralOption(Name, PassName(Name), Description);
}

int main(int Argc, char *Argv[]) {
  loadPassesList();

  revng::InitRevng X(Argc, Argv, "", { &ThisToolCategory, &ModelPassCategory });

  llvm::ExitOnError ExitOnError;
  auto ParsedModel = TupleTree<model::Binary>::fromFileOrSTDIN(InputFilename);
  auto MaybeModel = ExitOnError(std::move(ParsedModel));

  for (const PassName &PassName : PassesList) {
    const RegisteredModelPass *Registered = RegisterModelPass::get(PassName);
    if (Registered == nullptr) {
      ExitOnError(revng::createError("Pass not found: " + PassName));
    }

    // Run pass
    Registered->Pass(MaybeModel);
  }

  // Serialize
  ExitOnError(MaybeModel.toFile(OutputFilename));
}
