#pragma once

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include <string>

#include "revng/PTML/Tag.h"
#include "revng/Pipeline/Location.h"
#include "revng/Support/BasicBlockID.h"

class MetaAddress;
namespace model {
class Binary;
}
namespace yield {
class Function;
}

namespace ptml {
class MarkupBuilder;
}

namespace yield::ptml {

std::string functionAssembly(const ::ptml::MarkupBuilder &B,
                             const yield::Function &InternalFunction,
                             const model::Binary &Binary);
std::string controlFlowNode(const ::ptml::MarkupBuilder &B,
                            const BasicBlockID &BasicBlock,
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
std::string functionNameDefinition(const ::ptml::MarkupBuilder &B,
                                   llvm::StringRef Location,
                                   const model::Binary &Binary,
                                   model::AssemblyNameBuilder &NameBuilder);

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
std::string functionLink(const ::ptml::MarkupBuilder &B,
                         llvm::StringRef Location,
                         const model::Binary &Binary,
                         model::AssemblyNameBuilder &NameBuilder);

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
std::string shallowFunctionLink(const ::ptml::MarkupBuilder &B,
                                llvm::StringRef Location,
                                const model::Binary &Binary,
                                model::AssemblyNameBuilder &NameBuilder);

} // namespace yield::ptml
