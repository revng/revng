#ifndef _DEBUG_H
#define _DEBUG_H

// Standard includes
#include <ostream>

// LLVM includes
#include "llvm/Support/Debug.h"

// TODO: use a dedicated namespace
extern bool DebuggingEnabled;
extern std::ostream& dbg;

bool isDebugFeatureEnabled(std::string Name);
void enableDebugFeature(std::string Name);
void disableDebugFeature(std::string Name);

#define DBG(feature, code) do {                                     \
    if (DebuggingEnabled && isDebugFeatureEnabled(feature)) {       \
      code;                                                         \
    }                                                               \
  } while (0)


#endif // _DEBUG_H
