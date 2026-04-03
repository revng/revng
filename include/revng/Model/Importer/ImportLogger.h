#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "revng/Model/Binary.h"
#include "revng/Support/Debug.h"

/// RAII class to emit statistics on what has been imported in the model between
/// creation and destruction
class ImportLogger {
private:
  TupleTree<model::Binary> &Binary;
  class Logger &Logger;
  std::optional<LoggerIndent> Indent;
  size_t TypeDefinitions = 0;
  size_t LocalFunctions = 0;
  size_t DynamicFunctions = 0;

public:
  ImportLogger(TupleTree<model::Binary> &Binary,
               class Logger &Logger,
               llvm::StringRef Path) :
    Binary(Binary), Logger(Logger) {
    revng_log(Logger, "Importing debug information in " << Path);

    TypeDefinitions = Binary->TypeDefinitions().size();
    LocalFunctions = Binary->Functions().size();
    DynamicFunctions = Binary->ImportedDynamicFunctions().size();

    Indent.emplace(Logger);
  }

  ~ImportLogger() {
    size_t NewTypeDefinitions = Binary->TypeDefinitions().size()
                                - TypeDefinitions;
    size_t NewLocalFunctions = Binary->Functions().size() - LocalFunctions;
    size_t NewDynamicFunctions = Binary->ImportedDynamicFunctions().size()
                                 - DynamicFunctions;

    revng_log(Logger,
              "We imported: "
                << NewTypeDefinitions << " new type definitions, "
                << NewLocalFunctions << " new local functions and "
                << NewDynamicFunctions << " new dynamic functions");
  }
};
