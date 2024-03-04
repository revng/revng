#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/ADT/RecursiveCoroutine.h"
#include "revng/ADT/UpcastablePointer.h"
#include "revng/Model/CommonTypeMethods.h"
#include "revng/Model/Type.h"
#include "revng/Model/TypeDefinition.h"

namespace model::filter {

constexpr auto Scalar = std::views::filter([](AnyType auto &&T) {
  return T.isScalar();
});

} // namespace model::filter
