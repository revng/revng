#ifndef _DEBUG_H
#define _DEBUG_H

// Standard includes
#include <functional>
#include <ostream>
#include <string>

// LLVM includes
#include "llvm/Support/Debug.h"

// TODO: use a dedicated namespace
extern bool DebuggingEnabled;
extern std::ostream& dbg;

bool isDebugFeatureEnabled(std::string Name);
void enableDebugFeature(std::string Name);
void disableDebugFeature(std::string Name);

/// Executes \p code only if \p feature is enabled
// TODO: switch to lambda
#define DBG(feature, code) do {                                     \
    if (DebuggingEnabled && isDebugFeatureEnabled(feature)) {       \
      code;                                                         \
    }                                                               \
  } while (0)

/// \brief Enables a debug feature and disables it when goes out of scope
class ScopedDebugFeature {
public:
  /// \param Name the name of the debugging feature
  /// \param Enable whether to actually enable it or not
  ScopedDebugFeature(std::string Name, bool Enable)
    : Name(Name), Enabled(Enable) {
    if (Enabled)
      enableDebugFeature(Name);
  }

  ~ScopedDebugFeature() {
    if (Enabled)
      disableDebugFeature(Name);
  }

private:
  std::string Name;
  bool Enabled;
};

#endif // _DEBUG_H
