// Standard includes
#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

// Local includes
#include "debug.h"

bool DebuggingEnabled = false;
std::ostream& dbg(std::cerr);
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
