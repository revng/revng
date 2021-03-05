#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/ABIAnalyses/Generated/DeadRegisterArgumentsOfFunction.h"
#include "revng/ABIAnalyses/Generated/DeadReturnValuesOfFunctionCall.h"
#include "revng/ABIAnalyses/Generated/RegisterArgumentsOfFunctionCall.h"
#include "revng/ABIAnalyses/Generated/UsedArgumentsOfFunction.h"
#include "revng/ABIAnalyses/Generated/UsedReturnValuesOfFunction.h"
#include "revng/ABIAnalyses/Generated/UsedReturnValuesOfFunctionCall.h"

namespace ABIAnalyses {
namespace DeadRegisterArgumentsOfFunction {
using namespace llvm;

DenseMap<const GlobalVariable *, State>
analyze(const BasicBlock *FunctionEntry, const GeneratedCodeBasicInfo &GCBI);

} // namespace DeadRegisterArgumentsOfFunction

namespace DeadReturnValuesOfFunctionCall {
using namespace llvm;

DenseMap<const GlobalVariable *, State>
analyze(const BasicBlock *CallSiteBlock, const GeneratedCodeBasicInfo &GCBI);

} // namespace DeadReturnValuesOfFunctionCall

namespace RegisterArgumentsOfFunctionCall {
using namespace llvm;

DenseMap<const GlobalVariable *, State>
analyze(const BasicBlock *CallSiteBlock, const GeneratedCodeBasicInfo &GCBI);

} // namespace RegisterArgumentsOfFunctionCall

namespace UsedArgumentsOfFunction {
using namespace llvm;

DenseMap<const GlobalVariable *, State>
analyze(const BasicBlock *FunctionEntry, const GeneratedCodeBasicInfo &GCBI);

} // namespace UsedArgumentsOfFunction

namespace UsedReturnValuesOfFunction {
using namespace llvm;

DenseMap<const GlobalVariable *, State>
analyze(const BasicBlock *ReturnBlock, const GeneratedCodeBasicInfo &GCBI);
} // namespace UsedReturnValuesOfFunction

namespace UsedReturnValuesOfFunctionCall {
using namespace llvm;

DenseMap<const GlobalVariable *, State>
analyze(const BasicBlock *CallSiteBlock, const GeneratedCodeBasicInfo &GCBI);
} // namespace UsedReturnValuesOfFunctionCall

} // namespace ABIAnalyses
