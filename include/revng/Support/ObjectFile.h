#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <concepts>

#include "llvm/Object/ELFObjectFile.h"

template<typename T>
concept IsELFObjectFile = std::derived_from<T, llvm::object::ELFObjectFileBase>;
