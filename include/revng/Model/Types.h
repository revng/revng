#pragma once

//
// This file is distributed under the MIT License. See LICENSE.mit for details.
//

#include <algorithm>
#include <cctype>
#include <compare>
#include <optional>
#include <string>

#include "revng/ADT/SortedVector.h"
#include "revng/ADT/UpcastablePointer.h"
#include "revng/ADT/UpcastablePointer/YAMLTraits.h"
#include "revng/Model/CABIFunctionType.h"
#include "revng/Model/EnumType.h"
#include "revng/Model/PrimitiveType.h"
#include "revng/Model/RawFunctionType.h"
#include "revng/Model/StructType.h"
#include "revng/Model/Type.h"
#include "revng/Model/TypedefType.h"
#include "revng/Model/UnionType.h"
#include "revng/Support/MetaAddress.h"
#include "revng/Support/MetaAddress/YAMLTraits.h"

namespace model {
class VerifyHelper;
} // end namespace model
