#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <memory>

#include "llvm/Support/CommandLine.h"

#include "revng/Model/Binary.h"

namespace mlir::clift {

struct ModelOptionType : std::shared_ptr<const model::Binary> {
  using shared_ptr::shared_ptr;
};

} // namespace mlir::clift

template<>
class llvm::cl::parser<mlir::clift::ModelOptionType>
  : public basic_parser<mlir::clift::ModelOptionType> {
public:
  using basic_parser::basic_parser;

  bool parse(llvm::cl::Option &O,
             const llvm::StringRef ArgName,
             const llvm::StringRef ArgValue,
             mlir::clift::ModelOptionType &Value);

  StringRef getValueName() const override { return "<model path>"; }

  static void print(raw_ostream &os,
                    const mlir::clift::ModelOptionType &Value) {
    os << "<model::Binary>";
  }

  void printOptionDiff(const Option &O,
                       const mlir::clift::ModelOptionType &Value,
                       const OptVal &Default,
                       size_t GlobalWidth) const;
};
