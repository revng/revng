//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Support/Debug.h"

#include "revng-c/mlir/Dialect/Clift/Transforms/ModelOption.h"

using mlir::clift::ModelOptionType;
using TupleTreeType = TupleTree<model::Binary>;
using ParserType = llvm::cl::parser<ModelOptionType>;

bool ParserType::parse(llvm::cl::Option &O,
                       const llvm::StringRef ArgName,
                       const llvm::StringRef ArgValue,
                       ModelOptionType &Value) {
  auto MaybeModel = TupleTreeType::fromFile(ArgValue);

  if (not MaybeModel) {
    return O.error("Failed to parse model: " + consumeToString(MaybeModel));
  }

  auto Shared = std::make_shared<TupleTreeType>(std::move(*MaybeModel));
  Value = ModelOptionType(std::move(Shared), Shared->get());

  return false;
}

void ParserType::printOptionDiff(const Option &O,
                                 const ModelOptionType &Value,
                                 const OptVal &Default,
                                 const size_t GlobalWidth) const {
  printOptionName(O, GlobalWidth);
  outs() << "[=<model path>]";
}
