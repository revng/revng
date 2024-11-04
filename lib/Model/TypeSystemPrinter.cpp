//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"

#include "revng/Model/Binary.h"
#include "revng/Model/CABIFunctionDefinition.h"
#include "revng/Model/RawFunctionDefinition.h"
#include "revng/Model/StructDefinition.h"
#include "revng/Model/TypeDefinitionKind.h"
#include "revng/Model/TypeSystemPrinter.h"
#include "revng/Model/TypedefDefinition.h"
#include "revng/Model/UnionDefinition.h"
#include "revng/Support/Assert.h"

using llvm::cast;
using llvm::dyn_cast;
using llvm::isa;
using model::CABIFunctionDefinition;
using model::RawFunctionDefinition;
using model::StructDefinition;
using model::TypedefDefinition;
using model::UnionDefinition;
using std::to_string;

using FieldList = llvm::SmallVector<const model::Type *, 16>;

static constexpr const char *TableOpts = "border='0' cellborder='1' "
                                         "cellspacing='0' cellpadding='0'";
static constexpr const char *PaddingOpts = "cellpadding='10'";

static constexpr const char *Green = "\"#8DB596\"";
static constexpr const char *Red = "\"#DC7878\"";
static constexpr const char *Blue = "\"#93ABD3\"";
static constexpr const char *Orange = "\"#EEEE00\"";
static constexpr const char *Purple = "\"#C689C6\"";
static constexpr const char *Pink = "\"#FF99CC\"";
static constexpr const char *Grey = "\"#CCCCCC\"";
static constexpr const char *White = "\"white\"";

/// Background and border color for records of a given TypeDefinitionKind
static llvm::StringRef getColor(model::TypeDefinitionKind::Values K) {
  if (K == model::TypeDefinitionKind::UnionDefinition)
    return Red;
  else if (K == model::TypeDefinitionKind::CABIFunctionDefinition
           or K == model::TypeDefinitionKind::RawFunctionDefinition)
    return Green;
  else if (K == model::TypeDefinitionKind::StructDefinition)
    return Blue;

  return Grey;
}

/// Cell with inner padding, a colored background and a white border
static void headerCell(llvm::raw_ostream &Out,
                       llvm::StringRef Color,
                       llvm::StringRef Content) {
  Out << "<TD " << PaddingOpts << " color=" << White << " bgcolor=" << Color
      << ">" << Content << "</TD>";
}

/// Cell with inner padding and possibly a port identifier
static void paddedCell(llvm::raw_ostream &Out,
                       llvm::StringRef Content,
                       std::optional<size_t> Port = {}) {
  Out << "<TD " << PaddingOpts;
  if (Port)
    Out << " PORT='P" << to_string(Port.value()) << "'";
  Out << ">" << Content << "</TD>";
}

/// Collect an ordered list of the subtypes in a type (e.g. field, return
/// values, arguments ...)
static FieldList collectFields(const model::TypeDefinition *T) {
  FieldList Fields;

  if (auto *Struct = llvm::dyn_cast<model::StructDefinition>(T)) {
    for (auto &Field : Struct->Fields())
      Fields.push_back(Field.Type().get());

  } else if (auto *Union = llvm::dyn_cast<model::UnionDefinition>(T)) {
    for (auto &Field : Union->Fields())
      Fields.push_back(Field.Type().get());

  } else if (auto *CABI = llvm::dyn_cast<model::CABIFunctionDefinition>(T)) {
    Fields.push_back(CABI->ReturnType().get());
    for (auto &Field : CABI->Arguments())
      Fields.push_back(Field.Type().get());

  } else if (auto *RawFunc = llvm::dyn_cast<model::RawFunctionDefinition>(T)) {
    for (auto &Field : RawFunc->ReturnValues())
      Fields.push_back(Field.Type().get());

    for (auto &Field : RawFunc->Arguments())
      Fields.push_back(Field.Type().get());

    if (not RawFunc->StackArgumentsType().isEmpty())
      Fields.push_back(RawFunc->StackArgumentsType().get());
  } else if (auto *Typedef = llvm::dyn_cast<model::TypedefDefinition>(T)) {
    Fields.push_back(Typedef->UnderlyingType().get());
  }

  return Fields;
}

TypeSystemPrinter::TypeSystemPrinter(llvm::raw_ostream &Out,
                                     const model::Binary &Binary,
                                     bool OrthoEdges) :
  Out(Out), Binary(Binary), NameBuilder(Binary) {
  Out << "digraph TypeGraph {\n";
  if (OrthoEdges)
    Out << "splines=ortho;\n";
  Out << "node [shape=none, margin=0];\n";
  Out << "graph [fontname=Courier];\n";
  Out << "node [fontname=Courier];\n";
  Out << "edge [fontname=Courier];\n";
}

TypeSystemPrinter::~TypeSystemPrinter() {
  Out << "}\n";
  Out.flush();
}

/// Build a C-like string for a given Type
// TODO: replace this with a call to `getNamedCInstance` once the two repos
//       are merged together.
std::string TypeSystemPrinter::buildFieldName(const model::Type &Type,
                                              std::string &&Prefix,
                                              std::string &&Suffix) {
  if (const auto *Array = llvm::dyn_cast<model::ArrayType>(&Type)) {
    revng_assert(!Array->IsConst(),
                 "Const arrays are not supported by this serializer.");

    Suffix = "[" + std::to_string(Array->ElementCount()) + "]"
             + std::move(Suffix);
    return buildFieldName(*Array->ElementType(),
                          std::move(Prefix),
                          std::move(Suffix));

  } else if (const auto *D = llvm::dyn_cast<model::DefinedType>(&Type)) {
    std::string Result = std::move(Prefix);
    if (!Result.empty() && Result.back() != '*')
      Result += ' ';

    return Result += (NameBuilder.name(D->unwrap()) + Suffix).str();

  } else if (const auto *P = llvm::dyn_cast<model::PointerType>(&Type)) {
    return buildFieldName(*P->PointeeType(),
                          std::move(Prefix += P->IsConst() ? "* const" : "*"),
                          std::move(Suffix));

  } else if (const auto *P = llvm::dyn_cast<model::PrimitiveType>(&Type)) {
    std::string Result = std::move(Prefix);
    if (!Result.empty() && Result.at(Result.size() - 1) != '*')
      Result += ' ';

    return Result += P->getCName() + Suffix;

  } else {
    revng_abort("Unsupported type.");
  }
}

/// Add a row in a struct table
static void addStructField(llvm::raw_ostream &Out,
                           size_t Offset,
                           size_t Size,
                           llvm::StringRef Content,
                           std::optional<size_t> Port = {}) {
  Out << "<TR>";
  paddedCell(Out, to_string(Offset));
  paddedCell(Out, to_string(Size));
  paddedCell(Out, Content, Port);
  Out << "</TR>";
}

/// Generate the inner table of a struct type
void TypeSystemPrinter::dumpStructFields(llvm::raw_ostream &Out,
                                         const model::StructDefinition *T) {
  if (T->Fields().size() == 0) {
    Out << "<TR><TD></TD></TR>";
    return;
  }

  // Header
  llvm::StringRef Color = getColor(model::TypeDefinitionKind::StructDefinition);
  Out << "<TR>";
  headerCell(Out, Color, "Offset");
  headerCell(Out, Color, "Size");
  headerCell(Out, Color, "Name");
  Out << "</TR>";

  // Struct fields are stacked vertically
  uint64_t LastOffset = 0;
  for (auto [Index, Field] : llvm::enumerate(T->Fields())) {
    // Check if there's padding to be added before this field
    if (Field.Offset() > LastOffset)
      addStructField(Out, LastOffset, Field.Offset() - LastOffset, "padding");

    auto Name = buildFieldName(*Field.Type());
    uint64_t Size = Field.Type()->size().value_or(0);
    addStructField(Out, Field.Offset(), Size, Name, Index);

    LastOffset += Field.Offset() + Size;
  }

  // Check if there's trailing padding
  auto StructSize = T->size().value_or(0);
  if (StructSize > LastOffset)
    addStructField(Out, LastOffset, StructSize - LastOffset, "padding");
}

/// Generate the inner table of a union type
void TypeSystemPrinter::dumpUnionFields(llvm::raw_ostream &Out,
                                        const model::UnionDefinition *T) {
  if (T->Fields().size() == 0) {
    Out << "<TR><TD></TD></TR>";
    return;
  }

  Out << "<TR>";

  // Union fields are disposed horizontally
  for (auto [Index, Field] : llvm::enumerate(T->Fields())) {
    auto Name = buildFieldName(*Field.Type());
    const auto Size = Field.Type()->size().value_or(0);
    paddedCell(Out, Name + "  (size: " + to_string(Size) + ")", Index);
  }
  Out << "</TR>";
}

/// Generate the inner table of a function type
void TypeSystemPrinter::dumpFunctionType(llvm::raw_ostream &Out,
                                         const model::TypeDefinition *T) {
  llvm::SmallVector<const model::Type *, 8> ReturnTypes;
  llvm::SmallVector<const model::Type *, 8> Arguments;

  // Collect arguments and return types
  if (auto *RawFunc = dyn_cast<RawFunctionDefinition>(T)) {
    for (auto &RetTy : RawFunc->ReturnValues())
      ReturnTypes.push_back(RetTy.Type().get());

    for (auto &ArgTy : RawFunc->Arguments())
      Arguments.push_back(ArgTy.Type().get());

    if (not RawFunc->StackArgumentsType().isEmpty())
      Arguments.push_back(RawFunc->StackArgumentsType().get());

  } else if (auto *CABIFunc = dyn_cast<CABIFunctionDefinition>(T)) {
    ReturnTypes.push_back(CABIFunc->ReturnType().get());

    for (auto &ArgTy : CABIFunc->Arguments())
      Arguments.push_back(ArgTy.Type().get());
  }

  // Inner table that divides return types and arguments
  Out << "<TR><TD><TABLE " << TableOpts << ">";

  // Header
  llvm::StringRef Color = getColor(T->Kind());
  Out << "<TR>";
  headerCell(Out, Color, "Return Types");
  headerCell(Out, Color, "Arguments");
  Out << "</TR>";

  // Second row of the inner table (actual types)
  Out << "<TR>";

  size_t CurPort = 0;

  // Return types are disposed horizontally in a dedicated table
  Out << "<TD><TABLE " << TableOpts << "><TR>";
  if (ReturnTypes.empty()) {
    paddedCell(Out, "void");
    CurPort++;
  } else {
    for (auto Field : ReturnTypes)
      paddedCell(Out, buildFieldName(*Field), CurPort++);
  }
  Out << "</TR></TABLE></TD>";

  // Arguments types are disposed horizontally in a dedicated table
  Out << "<TD><TABLE " << TableOpts << "><TR>";
  if (Arguments.empty()) {
    Out << "<TD></TD>";
  } else {
    for (auto Field : Arguments)
      paddedCell(Out, buildFieldName(*Field), CurPort++);
  }
  Out << "</TR></TABLE></TD>";

  // End of second row
  Out << "</TR>";

  // End of inner table
  Out << "</TABLE></TD></TR>";
}

/// Generate the inner content of a Typedef node
void TypeSystemPrinter::dumpTypedefUnderlying(llvm::raw_ostream &Out,
                                              const model::TypedefDefinition
                                                *T) {
  Out << "<TR>";
  paddedCell(Out, buildFieldName(*T->UnderlyingType()), 0);
  Out << "</TR>";
}

void TypeSystemPrinter::dumpTypeNode(const model::TypeDefinition *T,
                                     int NodeID) {
  // Print the name of the node
  Out << "node_" << to_string(NodeID) << "[";

  // Choose the node's border color
  llvm::StringRef Color = getColor(T->Kind());
  Out << "color=" << Color << ", ";

  // Start of HTML-style label
  Out << "label= < <TABLE " << TableOpts << ">";

  // Print the name of the type on top
  Out << "<TR><TD bgcolor=" << Color << " " << PaddingOpts << " PORT='TOP'><B>"
      << NameBuilder.name(*T)
      << "</B>  (size: " << to_string(T->trySize().value_or(0))
      << ")</TD></TR>";

  // Print fields in a table
  Out << "<TR><TD><TABLE " << TableOpts << "> ";
  if (auto *StructT = dyn_cast<StructDefinition>(T))
    dumpStructFields(Out, StructT);
  else if (auto *UnionT = dyn_cast<UnionDefinition>(T))
    dumpUnionFields(Out, UnionT);
  else if (isa<RawFunctionDefinition>(T) or isa<CABIFunctionDefinition>(T))
    dumpFunctionType(Out, T);
  else if (auto *Typedef = dyn_cast<TypedefDefinition>(T))
    dumpTypedefUnderlying(Out, Typedef);
  else
    Out << "<TR><TD>Unhandled Type</TD></TR>";
  Out << "</TABLE></TD></TR>";

  // End of label
  Out << "</TABLE> >];\n";
}

void TypeSystemPrinter::addEdge(int SrcID, int SrcPort, int DstID) {
  Out << "node_" << to_string(SrcID) << ":<P" << to_string(SrcPort) << ">";
  Out << " -> ";
  Out << "node_" << to_string(DstID);
  Out << ":<TOP>;\n";
}

struct FieldEdge {
  std::string Label;
  const model::TypeDefinition *Destination;
  bool IsPointer;
};
static RecursiveCoroutine<FieldEdge>
buildFieldEdgeLabel(const model::Type &Type,
                    std::string &&Current = {},
                    bool IsPointer = false) {
  if (const auto *Array = llvm::dyn_cast<model::ArrayType>(&Type)) {
    if (!Current.empty())
      Current += ",\\n";
    Current += "Array[" + std::to_string(Array->ElementCount()) + "]";
    rc_return buildFieldEdgeLabel(*Array->ElementType(),
                                  std::move(Current),
                                  false);

  } else if (const auto *D = llvm::dyn_cast<model::DefinedType>(&Type)) {
    rc_return{ .Label = std::move(Current),
               .Destination = &D->unwrap(),
               .IsPointer = IsPointer };

  } else if (const auto *P = llvm::dyn_cast<model::PointerType>(&Type)) {
    if (!Current.empty())
      Current += ",\\n";
    Current += "Pointer (" + std::to_string(P->PointerSize()) + " bytes)";
    rc_return buildFieldEdgeLabel(*P->PointeeType(), std::move(Current), true);

  } else if (const auto *P = llvm::dyn_cast<model::PrimitiveType>(&Type)) {
    rc_return{ .Label = std::move(Current),
               .Destination = nullptr,
               .IsPointer = IsPointer };

  } else {
    revng_abort("Unsupported type.");
  }
}

void TypeSystemPrinter::addFieldEdge(std::string &&Label,
                                     bool IsPointer,
                                     int SrcID,
                                     int SrcPort,
                                     int DstID) {
  // Edge
  Out << "node_" << to_string(SrcID) << ":<P" << to_string(SrcPort) << ">";
  Out << " -> ";
  Out << "node_" << to_string(DstID) << ":<TOP>";

  // Label
  Out << "[label=\"" << std::move(Label) << "\"";

  // Style
  if (IsPointer)
    Out << ", style=dotted";

  Out << "];\n";
}

void TypeSystemPrinter::print(const model::TypeDefinition &T) {
  // Don't repeat nodes
  if (Visited.contains(&T))
    return;

  llvm::SmallVector<const model::TypeDefinition *, 16> ToVisit = { &T };

  auto EmitNode = [this](const model::TypeDefinition *TypeToEmit) {
    dumpTypeNode(TypeToEmit, NextID);
    NodesMap.insert({ TypeToEmit, NextID });
    NextID++;
  };

  // Emit the root
  EmitNode(&T);

  while (not ToVisit.empty()) {
    const model::TypeDefinition *CurType = ToVisit.pop_back_val();
    if (Visited.contains(CurType))
      continue;

    uint64_t CurID = NodesMap.at(CurType);

    // Collect all the successors
    FieldList Fields = collectFields(CurType);
    for (auto [Index, Field] : llvm::enumerate(Fields)) {
      FieldEdge Edge = buildFieldEdgeLabel(*Field);
      auto [Label, DefinitionPointer, IsPointer] = Edge;

      // Don't add edges for primitive types, as they would pollute the graph
      // and add no information regarding the type system structure
      if (DefinitionPointer == nullptr)
        continue;

      uint64_t SuccID;
      auto It = NodesMap.find(DefinitionPointer);
      if (It != NodesMap.end()) {
        // If a node already exists for the target type, use that
        SuccID = It->second;
      } else {
        // If the node does not already exist, create a new one
        SuccID = NextID;
        EmitNode(DefinitionPointer);
      }

      // Add an edge to the type referenced by the current field.
      // Since we have created the target type if it does not exist, and
      // the source node was either the root node or a successor of a
      // previously visited node, we are sure that both the source and the
      // destination of this edge have already been created.
      addFieldEdge(std::move(Label), IsPointer, CurID, Index, SuccID);

      // Push the field's type to the visit stack
      ToVisit.push_back(DefinitionPointer);
    }

    // Mark this Type as visited: the node has been emitted, as well as
    // all of its outgoing edges and their respective target nodes.
    Visited.insert(CurType);
  }
}

void TypeSystemPrinter::dumpFunctionNode(const model::Function &F, int NodeID) {
  // Print the name of the node
  Out << "node_" << to_string(NodeID) << "[";

  // Choose the node's border color
  llvm::StringRef Color = Purple;
  Out << "color=" << Color << ", ";

  // Start of HTML-style label
  Out << "label= < <TABLE " << TableOpts << ">";

  // Print the name of the function on top
  Out << "<TR><TD bgcolor=" << Color << " " << PaddingOpts << "><B>"
      << NameBuilder.name(F) << "()</B></TD></TR>";

  // Print connected types in a table
  Out << "<TR><TD><TABLE " << TableOpts << "> ";
  // Header
  Out << "<TR>";
  headerCell(Out, Color, "Prototype");
  headerCell(Out, Color, "StackType");
  Out << "</TR>";

  // Second row of the inner table (actual types)
  Out << "<TR>";

  if (const model::TypeDefinition *Prototype = F.prototype())
    paddedCell(Out, NameBuilder.name(*Prototype), /*port=*/0);
  else
    Out << "<TD></TD>";

  if (const model::StructDefinition *StackFrame = F.stackFrameType())
    paddedCell(Out, NameBuilder.name(*StackFrame), /*port=*/1);
  else
    Out << "<TD></TD>";

  Out << "</TR>";

  // End of inner table
  Out << "</TABLE></TD></TR>";

  // End of label
  Out << "</TABLE> >];\n";
}

void TypeSystemPrinter::print(const model::Function &F) {
  // Node corresponding to the function
  uint64_t FunctionNodeID = NextID;
  dumpFunctionNode(F, FunctionNodeID);
  NextID++;

  // Node of prototype type, if present
  if (const model::TypeDefinition *Prototype = F.prototype()) {
    print(*Prototype);

    // Edges
    uint64_t PrototypeNodeID = NodesMap.at(Prototype);
    addEdge(FunctionNodeID, 0, PrototypeNodeID);
  }

  // Node of the stack type, if present
  if (const model::StructDefinition *StackFrame = F.stackFrameType()) {
    print(*StackFrame);

    // Edges
    uint64_t StackNodeID = NodesMap.at(StackFrame);
    addEdge(FunctionNodeID, 1, StackNodeID);
  }
}

void TypeSystemPrinter::dumpFunctionNode(const model::DynamicFunction &F,
                                         int NodeID) {
  // Print the name of the node
  Out << "node_" << to_string(NodeID) << "[";

  // Choose the node's border color
  llvm::StringRef Color = Pink;
  Out << "color=" << Color << ", ";

  // Start of HTML-style label
  Out << "label= < <TABLE " << TableOpts << ">";

  // Print the name of the function on top
  Out << "<TR><TD bgcolor=" << Color << " " << PaddingOpts << "><B>"
      << NameBuilder.name(F) << "()</B></TD></TR>";

  // Print connected types in a table
  Out << "<TR><TD><TABLE " << TableOpts << "> ";
  // Header
  Out << "<TR>";
  headerCell(Out, Color, "Prototype");
  Out << "</TR>";

  // Second row of the inner table (actual types)
  Out << "<TR>";

  if (const model::TypeDefinition *Prototype = F.prototype())
    paddedCell(Out, NameBuilder.name(*Prototype), /*port=*/0);
  else
    Out << "<TD></TD>";

  Out << "</TR>";

  // End of inner table
  Out << "</TABLE></TD></TR>";

  // End of label
  Out << "</TABLE> >];\n";
}

void TypeSystemPrinter::print(const model::DynamicFunction &F) {
  // Node corresponding to the function
  uint64_t FunctionNodeID = NextID;
  dumpFunctionNode(F, FunctionNodeID);
  NextID++;

  // Node of the prototype type, if present
  if (const model::TypeDefinition *Prototype = F.prototype()) {
    print(*Prototype);

    // Edges
    uint64_t PrototypeNodeID = NodesMap.at(Prototype);
    addEdge(FunctionNodeID, 0, PrototypeNodeID);
  }
}

void TypeSystemPrinter::dumpSegmentNode(const model::Segment &S, int NodeID) {
  // Print the name of the node
  Out << "node_" << to_string(NodeID) << "[";

  // Choose the node's border color
  llvm::StringRef Color = Orange;
  Out << "color=" << Color << ", ";

  // Start of HTML-style label
  Out << "label= < <TABLE " << TableOpts << ">";

  // Print the name of the function on top
  Out << "<TR><TD bgcolor=" << Color << " " << PaddingOpts << "><B>"
      << NameBuilder.name(S) << "()</B></TD></TR>";

  // Print connected types in a table
  Out << "<TR><TD><TABLE " << TableOpts << "> ";
  // Header
  Out << "<TR>";
  headerCell(Out, Color, "Type");
  Out << "</TR>";

  // Second row of the inner table (actual types)
  Out << "<TR>";

  if (const model::StructDefinition *Type = S.type())
    paddedCell(Out, NameBuilder.name(*Type), /*port=*/0);
  else
    Out << "<TD></TD>";

  Out << "</TR>";

  // End of inner table
  Out << "</TABLE></TD></TR>";

  // End of label
  Out << "</TABLE> >];\n";
}

void TypeSystemPrinter::print(const model::Segment &S) {
  // Node corresponding to the function
  uint64_t SegmentNodeID = NextID;
  dumpSegmentNode(S, SegmentNodeID);
  NextID++;

  // Node of the prototype type, if present
  if (const model::StructDefinition *Type = S.type()) {
    print(*Type);

    // Edges
    uint64_t PrototypeNodeID = NodesMap.at(Type);
    addEdge(SegmentNodeID, 0, PrototypeNodeID);
  }
}

void TypeSystemPrinter::print() {
  // Print all functions and related types
  for (auto &F : Binary.Functions())
    print(F);

  // Print all dynamic functions and related types
  for (auto &F : Binary.ImportedDynamicFunctions())
    print(F);

  // Print all the segments and related types
  for (auto &S : Binary.Segments())
    print(S);

  // Print remaining types, if any
  for (auto &T : Binary.TypeDefinitions())
    if (!NodesMap.contains(T.get()))
      print(*T);
}
