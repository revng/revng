#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/Support/Error.h"

#include "revng/ABI/RegisterState.h"
#include "revng/Model/Binary.h"

#include "MFIGraphs/RegisterArgumentsOfFunctionCall.h"
#include "MFIGraphs/UsedArgumentsOfFunction.h"
#include "MFIGraphs/UsedReturnValuesOfFunction.h"
#include "MFIGraphs/UsedReturnValuesOfFunctionCall.h"

namespace ABIAnalyses {

namespace RegisterArgumentsOfFunctionCall {
using namespace llvm;

std::map<const GlobalVariable *, abi::RegisterState::Values>
analyze(const BasicBlock *CallSiteBlock, const GeneratedCodeBasicInfo &GCBI);

} // namespace RegisterArgumentsOfFunctionCall

namespace UsedArgumentsOfFunction {
using namespace llvm;

std::map<const GlobalVariable *, abi::RegisterState::Values>
analyze(const BasicBlock *FunctionEntry, const GeneratedCodeBasicInfo &GCBI);

} // namespace UsedArgumentsOfFunction

namespace UsedReturnValuesOfFunction {
using namespace llvm;

std::map<const GlobalVariable *, abi::RegisterState::Values>
analyze(const BasicBlock *ReturnBlock, const GeneratedCodeBasicInfo &GCBI);
} // namespace UsedReturnValuesOfFunction

namespace UsedReturnValuesOfFunctionCall {
using namespace llvm;

std::map<const GlobalVariable *, abi::RegisterState::Values>
analyze(const BasicBlock *CallSiteBlock, const GeneratedCodeBasicInfo &GCBI);
} // namespace UsedReturnValuesOfFunctionCall

} // namespace ABIAnalyses
