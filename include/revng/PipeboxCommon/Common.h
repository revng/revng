#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <vector>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include "revng/PipeboxCommon/ObjectID.h"

namespace revng::pypeline {

/// Type used for both incoming and outgoing requests
/// Each index maps to the i-th container passed to the 'run' function
using Request = std::vector<std::vector<const ObjectID *>>;
/// Type representing a path in the model, returned in a set when an analysis
/// runs
using ModelPath = std::string;
/// Type used to return the dependencies of objects produced by a pipe
/// The first index maps to the i-th container of the pipe
using ObjectDependencies = std::vector<
  std::vector<std::pair<ObjectID, ModelPath>>>;

/// Defines the access of a container when declaring a Pipe{,Run}Argument. In
/// PipeRuns there needs to be exactly one container with either Write or
/// ReadWrite as that will be the one where the model dependencies will be
/// tracked upon.
enum class Access {
  /// The requested container will only be read (request objects as a
  /// dependency). Note that specifying this allows the container to be used
  /// without `const`, this is intended but should really be used as a
  /// last-resort in situations where the container remains conceptually const
  /// but cannot be for performance reasons.
  Read,
  /// The requested container will be written (don't request objects as a
  /// dependency, the pipe will produce them). This is required when a pipe is
  /// the first to write to a container.
  Write,
  /// The requested container will be overwritten in-place (e.g. LLVM Pipe).
  /// This still requests objects as a dependency.
  ReadWrite,
  /// Automatically detect the access based on const-ness
  /// * const -> Read
  /// * non-const -> ReadWrite
  Auto,
};

template<ConstexprString N, ConstexprString HT, Access A = Access::Auto>
struct PipeArgument {
  static constexpr llvm::StringRef Name = N;
  static constexpr llvm::StringRef HelpText = HT;
  static constexpr Access Access = A;
};

template<typename T,
         ConstexprString N,
         ConstexprString HT,
         Access A = Access::Auto>
struct PipeRunArgument : public PipeArgument<N, HT, A> {
  using Type = T;
};

class Buffer {
private:
  llvm::SmallVector<char, 0> Vector;

public:
  template<typename... T>
  Buffer(T... Args) : Vector(std::forward<T>(Args)...) {}

  llvm::SmallVector<char, 0> &data() { return Vector; }

  // Read the contents
  llvm::ArrayRef<char> data() const { return { Vector.data(), Vector.size() }; }
};

} // namespace revng::pypeline
