#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

namespace llvm {
class Instruction;
}

class MetaAddress;

namespace revng {

std::optional<MetaAddress> tryExtractAddress(const llvm::Instruction &I);

} // namespace revng
