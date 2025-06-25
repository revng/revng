#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/TypeNames/ModelCBuilder.h"

namespace ptml {

class HeaderBuilder {
public:
  ModelCBuilder &B;

public:
  struct ConfigurationOptions {
    /// The piece of code to be inserted on top of the emitted header file
    /// (right after the normal includes).
    std::string PostIncludeSnippet = {};

    /// Sometimes you don't want to print everything. This lets you specify
    /// a set of functions that will be ignored by \ref functionPrototype.
    std::set<MetaAddress> FunctionsToOmit = {};
  };
  const ConfigurationOptions Configuration;

public:
  HeaderBuilder(ModelCBuilder &B, ConfigurationOptions &&Configuration) :
    B(B), Configuration(std::move(Configuration)) {}
  HeaderBuilder(ModelCBuilder &B) : B(B), Configuration() {}

public:
  /// Generate a C header containing a serialization of the type system,
  /// i.e. function prototypes, structs, unions, typedefs, and anything that
  /// resides in the model.
  bool printModelHeader();

  /// Generate a C header containing the declaration of each non-isolated
  /// function in a given LLVM IR module, i.e. QEMU helpers and revng helpers,
  /// whose prototype is not in the model. For helpers that return a struct, a
  /// new struct type will be defined and serialized on-the-fly.
  bool printHelpersHeader(const llvm::Module &Module);

  // TODO: Generate primitives header instead of just providing a hard-coded
  //       one in order to be able to hook it up to the location system,
  //       allowing the user to look a primitive definition up.
  // bool printPrimitivesHeader();
};

} // namespace ptml
