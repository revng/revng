//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/STLExtras.h"

#include "revng/CliftPipes/ConfigureCDialect.h"
#include "revng/Support/YAMLTraits.h"

using namespace clift;

namespace {

struct CDialectWrapper : CDialect {};

} // namespace

template<>
struct llvm::yaml::MappingTraits<CDialectWrapper> {
  static void mapping(IO &TheIO, CDialectWrapper &Value) {
    auto ToKebabCase = [](llvm::StringRef PascalCase) {
      std::string KebabCase;
      for (auto [I, C] : llvm::enumerate(PascalCase)) {
        char LowerCase = C;
        if ('A' <= C and C <= 'Z') {
          if (I != 0)
            KebabCase.push_back('-');

          LowerCase = 'a' + (C - 'A');
        }

        KebabCase.push_back(LowerCase);
      }
      return KebabCase;
    };

#define C_DIALECT_OPTION(Option) \
  TheIO.mapOptional(ToKebabCase(#Option).c_str(), Value.Option);

#include "revng/Support/CDialect.inc"
  }
};

static CDialect parseCDialect(llvm::StringRef Config) {
  if (Config.empty())
    return CDialect::Default;

  return llvm::cantFail(fromString<CDialectWrapper>(Config));
}

using Pipe = revng::pypeline::piperuns::ConfigureFunctionCDialect;

Pipe::ConfigureFunctionCDialect(const Model &Model,
                                llvm::StringRef Config,
                                llvm::StringRef DynamicConfig,
                                CliftFunctionContainer &ModuleContainer) :
  CliftFunctionMixin(ModuleContainer), Dialect(parseCDialect(DynamicConfig)) {
}

void Pipe::runOnCliftFunction(const model::Function &Model, FunctionOp Op) {
  if (Dialect != CDialect::Default)
    setCDialect(Op->getParentOfType<mlir::ModuleOp>(), Dialect);
}
