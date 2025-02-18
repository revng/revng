#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <map>
#include <memory>

#include "revng/Yield/Function.h"

namespace detail {
class DissassemblyHelperImpl;
}
namespace model {
class Function;
}
namespace efa {
class ControlFlowGraph;
}
class LLVMDisassemblerInterface;
class RawBinaryView;

class DissassemblyHelper {
private:
  std::unique_ptr<detail::DissassemblyHelperImpl> Internal;

public:
  explicit DissassemblyHelper();
  ~DissassemblyHelper();

  yield::Function disassemble(const model::Function &Function,
                              const efa::ControlFlowGraph &Metadata,
                              const RawBinaryView &BinaryView,
                              const model::Binary &Binary,
                              model::AssemblyNameBuilder &NameBuilder);

private:
  LLVMDisassemblerInterface &
  getDisassemblerFor(MetaAddressType::Values AddressType,
                     const model::DisassemblyConfiguration &);
};
