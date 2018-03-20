/// \file debug.cpp
/// \brief Implementation of the debug framework

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

// Standard includes
#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

// LLVM includes
#include "llvm/IR/Value.h"

// Local includes
#include "debug.h"

#ifndef NDEBUG
namespace llvm {
void Value::assertModuleIsMaterialized() const { }
}
#endif

bool DebuggingEnabled = false;
std::ostream &dbg(std::cerr);
static std::vector<std::string> DebugFeatures;

bool isDebugFeatureEnabled(std::string Name) {
  return std::find(DebugFeatures.begin(),
                   DebugFeatures.end(),
                   Name) != DebugFeatures.end();
}

void enableDebugFeature(std::string Name) {
  if (!isDebugFeatureEnabled(Name))
    DebugFeatures.push_back(Name);
}

void disableDebugFeature(std::string Name) {
  auto It = std::find(DebugFeatures.begin(),
                      DebugFeatures.end(),
                      Name);
  if (It != DebugFeatures.end())
    DebugFeatures.erase(It);
}
