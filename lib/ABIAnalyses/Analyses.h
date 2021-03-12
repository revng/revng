#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "Generated/DeadRegisterArgumentsOfFunction.h"
#include "Generated/DeadReturnValuesOfFunctionCall.h"
#include "Generated/RegisterArgumentsOfFunctionCall.h"
#include "Generated/UsedArgumentsOfFunction.h"
#include "Generated/UsedReturnValuesOfFunction.h"
#include "Generated/UsedReturnValuesOfFunctionCall.h"

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
