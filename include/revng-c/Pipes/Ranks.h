#pragma once

//
// Copyright (c) rev.ng Labs Srl. See LICENSE.md for details.
//

#include "revng/Model/Argument.h"
#include "revng/Model/Function.h"
#include "revng/Model/NamedTypedRegister.h"
#include "revng/Pipes/Ranks.h"
#include "revng/Support/YAMLTraits.h"

namespace revng::ranks {

using pipeline::defineRank;

using RK = model::NamedTypedRegister::Key;
using AK = model::Argument::Key;
inline const constexpr ConstexprString RFA("raw-function-argument");
inline const constexpr ConstexprString CFA("cabi-function-argument");
inline const constexpr ConstexprString RDFA("raw-dynamic-function-argument");
inline const constexpr ConstexprString CDFA("cabi-dynamic-function-argument");
inline auto RawFunctionArgument = defineRank<RFA, RK>(Function);
inline auto CABIFunctionArgument = defineRank<CFA, AK>(Function);
inline auto RawDynFunctionArgument = defineRank<RDFA, RK>(DynamicFunction);
inline auto CABIDynFunctionArgument = defineRank<CDFA, AK>(DynamicFunction);

inline const constexpr ConstexprString SV("special-variable");
inline auto SpecialVariable = defineRank<SV, std::string>(Function);
inline auto LocalVariable = defineRank<"local-variable", std::string>(Function);

inline auto HelperFunction = defineRank<"helper-function", std::string>(Binary);
inline auto HelperStruct = defineRank<"helper-struct", std::string>(Binary);
inline const constexpr ConstexprString HSF("helper-struct-field");
inline auto HelperStructField = defineRank<HSF, std::string>(HelperStruct);

} // namespace revng::ranks
