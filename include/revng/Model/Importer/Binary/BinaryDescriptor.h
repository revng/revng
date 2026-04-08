#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Path.h"

#include "revng/Model/BinaryIdentifier.h"
#include "revng/Support/CommandLine.h"

namespace llvm {
namespace object {
class ELFObjectFileBase;
class COFFObjectFile;
class MachOObjectFile;
} // namespace object
} // namespace llvm

template<typename T>
struct BinaryDescriptor {
public:
  using ObjectFileType = T;

public:
  // Note: this has to be non-const because certain Mach-O methods mutate the
  //       object.
  T &ObjectFile;

  /// Path a temporary file containing the binary.
  ///
  /// This path should never be used for binary-relative lookups (e.g., $ORIGIN
  /// or .debug).
  /// Use this only to pass it as an argument to external programs (e.g.,
  /// fetch-debug-info). In native code, you should use ObjectFile above.
  std::string FullPathForExternalTools;

  const model::BinaryReference Reference;

  /// This returns the path where the binary is supposed to be on the file
  /// system.
  ///
  /// Use this for binary-relative lookups (e.g., $ORIGIN or .debug) but do not
  /// try to open, there are no guarantees there's an actual file a this path.
  ///
  /// Also, this might be just a file name, a relative path or even empty.
  /// Use with caution.
  llvm::StringRef canonicalPath() const {
    if (Reference.isValid()) {
      return Reference.get()->CanonicalPath();
    } else if (not InputPath.empty()) {
      // Old pipeline only: use InputPath, a global variable set by hand in
      // Main.cpp
      // TODO: this should be dismissed along with the old pipeline
      return InputPath;
    } else {
      // TODO: this should be dismissed along with the old pipeline
      return llvm::sys::path::filename(FullPathForExternalTools);
    }
  }
};

using ELFBinary = BinaryDescriptor<llvm::object::ELFObjectFileBase>;
using COFFBinary = BinaryDescriptor<llvm::object::COFFObjectFile>;
using MachOBinary = BinaryDescriptor<llvm::object::MachOObjectFile>;
