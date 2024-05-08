#pragma once

//
// This file is distributed under the MIT License. See LICENSE.mit for details.
//

#include "revng/Storage/Path.h"

namespace revng {

/// This enum is used to allow specifying via cl::init if the path should have
/// stdin/stdout as the default value.
enum class PathInit {
  None,
  Dash
};

namespace detail {

/// Utility class similar to std::optional, to be used as a llvm::cl::opt.
/// The DashGenerator template parameter dictates the Path that will be used
/// when the path '-' or cl::init(PathInit::Dash) are used
template<FilePath (*DashGenerator)()>
struct PathImpl {
private:
  std::optional<revng::FilePath> Path;

public:
  PathImpl() = default;

  void setDash() { Path = DashGenerator(); }

  void parse(llvm::StringRef Value) {
    if (Value == "-")
      Path = DashGenerator();
    else
      Path = FilePath::fromLocalStorage(Value);
  }

  bool hasValue() const { return Path.has_value(); }

  const FilePath &operator*() const {
    revng_assert(Path.has_value());
    return *Path;
  }
};

/// cl::parser specialization to be used in conjunction with PathImpl, will call
/// PathImpl::parse given the input string from the command-line.
template<FilePath (*DashGenerator)()>
class PathImplParser : public llvm::cl::parser<PathImpl<DashGenerator>> {
public:
  using llvm::cl::parser<PathImpl<DashGenerator>>::parser;

  bool parse(llvm::cl::Option &Option,
             llvm::StringRef ArgName,
             const llvm::StringRef ArgValue,
             PathImpl<DashGenerator> &Val) {
    Val.parse(ArgValue);
    return false;
  }
};

template<FilePath (*DG)()>
using PathImplOpt = llvm::cl::opt<PathImpl<DG>, false, PathImplParser<DG>>;

} // namespace detail

// cl::opts for input and output revng::FilePaths
using InputPathOpt = detail::PathImplOpt<FilePath::stdin>;
using OutputPathOpt = detail::PathImplOpt<FilePath::stdout>;

} // namespace revng

namespace llvm::cl {

// These 'apply' specializations are needed to allow using cl::init(PathInit)
// with the {Input,Output}PathOpt.

template<>
inline void
apply<revng::InputPathOpt,
      initializer<revng::PathInit>>(revng::InputPathOpt *O,
                                    const initializer<revng::PathInit> &M) {
  if (M.Init == revng::PathInit::Dash)
    O->setDash();
}

template<>
inline void
apply<revng::OutputPathOpt,
      initializer<revng::PathInit>>(revng::OutputPathOpt *O,
                                    const initializer<revng::PathInit> &M) {
  if (M.Init == revng::PathInit::Dash)
    O->setDash();
}

} // namespace llvm::cl
