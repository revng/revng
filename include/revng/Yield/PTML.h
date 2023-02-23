#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <string>

#include "revng/Support/BasicBlockID.h"

class MetaAddress;
namespace model {
class Binary;
}
namespace yield {
class Function;
}

namespace yield::ptml {

std::string functionAssembly(const yield::Function &InternalFunction,
                             const model::Binary &Binary);
std::string controlFlowNode(const BasicBlockID &BasicBlockAddress,
                            const yield::Function &Function,
                            const model::Binary &Binary);

/// Emits a PTML containing a function name marked as a definition (it can be
/// jumped to).
///
/// For example,
/// ```html
///   <div data-token="call-graph.function"
///        data-location-definition="$FUNCTION_LOCATION">
///     $FUNCTION_NAME
///   </div>
/// ```
std::string functionNameDefinition(const MetaAddress &FunctionEntryPoint,
                                   const model::Binary &Binary);

/// Emits a PTML containing a function name marked as a reference
/// (it can be used to jump to any definition of the same function).
///
/// For example,
/// ```html
///   <div data-token="call-graph.function"
///        data-location-references="$FUNCTION_LOCATION">
///     $FUNCTION_NAME
///   </div>
/// ```
std::string functionLink(const MetaAddress &FunctionEntryPoint,
                         const model::Binary &Binary);

/// Emits a PTML containing a function name marked as a reference
/// \see functionLink
///
/// The only difference is that it uses a distinct token type to let clients
/// display it in a visually distinct way.
///
/// For example,
/// ```html
///   <div data-token="call-graph.shallow-function-link"
///        data-location-references="$FUNCTION_LOCATION">
///     $FUNCTION_NAME
///   </div>
/// ```
std::string shallowFunctionLink(const MetaAddress &FunctionEntryPoint,
                                const model::Binary &Binary);

} // namespace yield::ptml
