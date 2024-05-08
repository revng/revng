/// \file Main.cpp

//
// This file is distributed under the MIT License. See LICENSE.mit for details.
//

#include <string>

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"

#include "revng/Model/Pass/RegisterModelPass.h"
#include "revng/Model/Processing.h"
#include "revng/Model/ToolHelpers.h"
#include "revng/Support/InitRevng.h"

using namespace llvm;

static cl::OptionCategory ThisToolCategory("Tool options", "");
extern llvm::cl::OptionCategory ModelPassCategory;

static ModelOutputOptions<true> Options(ThisToolCategory);

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
  for (const auto &[Name, Description, _] : RegisterModelPass::passes())
    PassesList.getParser().addLiteralOption(Name, PassName(Name), Description);
}

int main(int Argc, char *Argv[]) {
  loadPassesList();

  revng::InitRevng X(Argc, Argv, "", { &ThisToolCategory, &ModelPassCategory });

  ExitOnError ExitOnError;
  auto MaybeModel = ExitOnError(ModelInModule::load(InputFilename));

  for (const PassName &PassName : PassesList) {
    const RegisterModelPass::ModelPass *Pass = RegisterModelPass::get(PassName);
    if (Pass == nullptr) {
      ExitOnError(llvm::createStringError(llvm::inconvertibleErrorCode(),
                                          "Pass not found: " + PassName));
    }

    // Run pass
    (*Pass)(MaybeModel.Model);
  }

  // Serialize
  auto OutputType = Options.getDesiredOutput(MaybeModel.hasModule());
  ExitOnError(MaybeModel.save(Options.getPath(), OutputType));
}
