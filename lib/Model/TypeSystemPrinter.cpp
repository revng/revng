//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"

#include "revng/Model/Binary.h"
#include "revng/Model/CABIFunctionType.h"
#include "revng/Model/QualifiedType.h"
#include "revng/Model/RawFunctionType.h"
#include "revng/Model/StructType.h"
#include "revng/Model/TypeKind.h"
#include "revng/Model/TypeSystemPrinter.h"
#include "revng/Model/TypedefType.h"
#include "revng/Model/UnionType.h"
#include "revng/Support/Assert.h"

using llvm::cast;
using llvm::dyn_cast;
using llvm::isa;
using model::CABIFunctionType;
using model::QualifiedType;
using model::RawFunctionType;
using model::StructType;
using model::TypedefType;
using model::UnionType;
using std::to_string;

using FieldList = llvm::SmallVector<QualifiedType, 16>;

static constexpr const char *TableOpts = "border='0' cellborder='1' "
                                         "cellspacing='0' cellpadding='0'";
static constexpr const char *PaddingOpts = "cellpadding='10'";

static constexpr const char *Green = "\"#8DB596\"";
static constexpr const char *LightGreen = "\"#D3EBCD\"";
static constexpr const char *Red = "\"#EC5858\"";
static constexpr const char *Blue = "\"#93ABD3\"";
static constexpr const char *Purple = "\"#C689C6\"";
static constexpr const char *Grey = "\"#7C3E66\"";
static constexpr const char *White = "\"white\"";

/// Background and border color for records of a given TypeKind
static llvm::StringRef getColor(model::TypeKind::Values K) {
  if (K == model::TypeKind::UnionType)
    return Red;
  else if (K == model::TypeKind::CABIFunctionType
           or K == model::TypeKind::RawFunctionType)
    return Green;
  else if (K == model::TypeKind::StructType)
    return Blue;

  return White;
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
static FieldList collectFields(const model::Type *T) {
  FieldList Fields;

  if (auto *Struct = llvm::dyn_cast<model::StructType>(T)) {
    for (auto &Field : Struct->Fields())
      Fields.push_back(Field.Type());

  } else if (auto *Union = llvm::dyn_cast<model::UnionType>(T)) {
    for (auto &Field : Union->Fields())
      Fields.push_back(Field.Type());

  } else if (auto *CABIFunc = llvm::dyn_cast<model::CABIFunctionType>(T)) {
    Fields.push_back(CABIFunc->ReturnType());
    for (auto &Field : CABIFunc->Arguments())
      Fields.push_back(Field.Type());

  } else if (auto *RawFunc = llvm::dyn_cast<model::RawFunctionType>(T)) {
    for (auto &Field : RawFunc->ReturnValues())
      Fields.push_back(Field.Type());

    if (Fields.empty())
      Fields.push_back({});

    for (auto &Field : RawFunc->Arguments())
      Fields.push_back(Field.Type());

    if (RawFunc->StackArgumentsType().UnqualifiedType().isValid())
      Fields.push_back(RawFunc->StackArgumentsType());
  } else if (auto *Typedef = llvm::dyn_cast<model::TypedefType>(T)) {
    Fields.push_back(Typedef->UnderlyingType());
  }

  return Fields;
}

TypeSystemPrinter::TypeSystemPrinter(llvm::raw_ostream &Out, bool OrthoEdges) :
  Out(Out) {
  Out << "digraph TypeGraph {\n";
  if (OrthoEdges)
    Out << "splines=ortho;\n";
  Out << "node [shape=none, margin=0];\n";
  Out << "graph [fontname=monospace];\n";
  Out << "node [fontname=monospace];\n";
  Out << "edge [fontname=monospace];\n";
}

TypeSystemPrinter::~TypeSystemPrinter() {
  Out << "}\n";
  Out.flush();
}

/// Build a C-like string for a given QualifiedType
static llvm::SmallString<32>
buildFieldName(const model::QualifiedType &FieldQT) {
  llvm::SmallString<32> FieldName;

  if (FieldQT.UnqualifiedType().isValid()) {
    FieldName += FieldQT.UnqualifiedType().get()->name();
    FieldName += " ";
  } else {
    FieldName += "void ";
  }

  for (auto &Q : FieldQT.Qualifiers()) {
    switch (Q.Kind()) {
    case model::QualifierKind::Pointer:
      FieldName += "*";
      break;
    case model::QualifierKind::Array:
      FieldName += "[" + to_string(Q.Size()) + "]";
      break;
    default:
      break;
    }
  }

  return FieldName;
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
static void
dumpStructFields(llvm::raw_ostream &Out, const model::StructType *T) {
  if (T->Fields().size() == 0) {
    Out << "<TR><TD></TD></TR>";
    return;
  }

  // Header
  auto Color = getColor(model::TypeKind::StructType);
  Out << "<TR>";
  headerCell(Out, Color, "Offset");
  headerCell(Out, Color, "Size");
  headerCell(Out, Color, "Name");
  Out << "</TR>";

  // Struct fields are stacked vertically
  uint64_t LastOffset = 0;
  for (auto FieldEnum : llvm::enumerate(T->Fields())) {
    const auto &Field = FieldEnum.value();
    const auto &FieldQT = Field.Type();
    const auto FieldOffset = Field.Offset();

    // Check if there's padding to be added before this field
    if (FieldOffset > LastOffset)
      addStructField(Out, LastOffset, FieldOffset - LastOffset, "padding");

    addStructField(Out,
                   Field.Offset(),
                   FieldQT.size().value_or(0),
                   buildFieldName(FieldQT),
                   FieldEnum.index());

    LastOffset += FieldOffset + FieldQT.size().value_or(0);
  }

  // Check if there's trailing padding
  auto StructSize = T->size().value_or(0);
  if (StructSize > LastOffset)
    addStructField(Out, LastOffset, StructSize - LastOffset, "padding");
}

/// Generate the inner table of a union type
static void dumpUnionFields(llvm::raw_ostream &Out, const model::UnionType *T) {
  if (T->Fields().size() == 0) {
    Out << "<TR><TD></TD></TR>";
    return;
  }

  Out << "<TR>";

  // Union fields are disposed horizontally
  for (auto FieldEnum : llvm::enumerate(T->Fields())) {
    const auto &Field = FieldEnum.value();
    const auto &FieldQT = Field.Type();
    const auto FieldSize = FieldQT.size().value_or(0);

    paddedCell(Out,
               (buildFieldName(FieldQT) + "  (size: " + to_string(FieldSize)
                + ")")
                 .str(),
               FieldEnum.index());
  }
  Out << "</TR>";
}

/// Generate the inner table of a function type
static void dumpFunctionType(llvm::raw_ostream &Out, const model::Type *T) {
  llvm::SmallVector<model::QualifiedType, 8> ReturnTypes;
  llvm::SmallVector<model::QualifiedType, 8> Arguments;

  // Collect arguments and return types
  if (auto *RawFunc = dyn_cast<RawFunctionType>(T)) {
    for (auto &RetTy : RawFunc->ReturnValues())
      ReturnTypes.push_back(RetTy.Type());

    for (auto &ArgTy : RawFunc->Arguments())
      Arguments.push_back(ArgTy.Type());

    if (RawFunc->StackArgumentsType().UnqualifiedType().isValid())
      Arguments.push_back(RawFunc->StackArgumentsType());

  } else if (auto *CABIFunc = dyn_cast<CABIFunctionType>(T)) {
    ReturnTypes.push_back(CABIFunc->ReturnType());

    for (auto &ArgTy : CABIFunc->Arguments())
      Arguments.push_back(ArgTy.Type());
  }

  // Inner table that divides return types and arguments
  Out << "<TR><TD><TABLE " << TableOpts << ">";

  // Header
  auto Color = getColor(T->Kind());
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
      paddedCell(Out, buildFieldName(Field), CurPort++);
  }
  Out << "</TR></TABLE></TD>";

  // Arguments types are disposed horizontally in a dedicated table
  Out << "<TD><TABLE " << TableOpts << "><TR>";
  if (Arguments.empty()) {
    Out << "<TD></TD>";
  } else {
    for (auto Field : Arguments)
      paddedCell(Out, buildFieldName(Field), CurPort++);
  }
  Out << "</TR></TABLE></TD>";

  // End of second row
  Out << "</TR>";

  // End of inner table
  Out << "</TABLE></TD></TR>";
}

/// Generate the inner content of a Typedef node
static void
dumpTypedefUnderlying(llvm::raw_ostream &Out, const model::TypedefType *T) {
  Out << "<TR>";
  paddedCell(Out, buildFieldName(T->UnderlyingType()), 0);
  Out << "</TR>";
}

void TypeSystemPrinter::dumpTypeNode(const model::Type *T, int NodeID) {
  // Print the name of the node
  Out << "node_" << to_string(NodeID) << "[";

  // Choose the node's border color
  auto Color = getColor(T->Kind());
  Out << "color=" << Color << ", ";

  // Start of HTML-style label
  Out << "label= < <TABLE " << TableOpts << ">";

  // Print the name of the type on top
  Out << "<TR><TD bgcolor=" << Color << " " << PaddingOpts << " PORT='TOP'><B>"
      << T->name() << "</B>  (size: " << to_string(T->size().value_or(0))
      << ")</TD></TR>";

  // Print fields in a table
  Out << "<TR><TD><TABLE " << TableOpts << "> ";
  if (auto *StructT = dyn_cast<StructType>(T))
    dumpStructFields(Out, StructT);
  else if (auto *UnionT = dyn_cast<UnionType>(T))
    dumpUnionFields(Out, UnionT);
  else if (isa<RawFunctionType>(T) or isa<CABIFunctionType>(T))
    dumpFunctionType(Out, T);
  else if (auto *Typedef = dyn_cast<TypedefType>(T))
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

void TypeSystemPrinter::addFieldEdge(const model::QualifiedType &QT,
                                     int SrcID,
                                     int SrcPort,
                                     int DstID) {
  // Edge
  Out << "node_" << to_string(SrcID) << ":<P" << to_string(SrcPort) << ">";
  Out << " -> ";
  Out << "node_" << to_string(DstID) << ":<TOP>";

  // Label
  Out << "[label=\"";

  const char *Prefix = "";
  for (auto &Qual : QT.Qualifiers()) {
    Out << Prefix;
    Prefix = ",\\n";
    switch (Qual.Kind()) {
    case model::QualifierKind::Array:
      Out << "Array[" << Qual.Size() << "]";
      break;
    case model::QualifierKind::Pointer:
      Out << "Pointer (size " << Qual.Size() << ")";
      break;
    default:
      break;
    }
  }

  Out << "\"";

  // Style
  if (QT.isPointer())
    Out << ", style=dotted";

  Out << "];\n";
}

void TypeSystemPrinter::print(const model::Type &T) {
  // Don't repeat nodes
  if (Visited.contains(&T))
    return;

  llvm::SmallVector<const model::Type *, 16> ToVisit = { &T };

  auto EmitNode = [this](const model::Type *TypeToEmit) {
    dumpTypeNode(TypeToEmit, NextID);
    NodesMap.insert({ TypeToEmit, NextID });
    NextID++;
  };

  // Emit the root
  EmitNode(&T);

  while (not ToVisit.empty()) {
    const model::Type *CurType = ToVisit.pop_back_val();
    if (Visited.contains(CurType))
      continue;

    uint64_t CurID = NodesMap.at(CurType);

    // Collect all the successors
    FieldList Fields = collectFields(CurType);
    for (auto Field : llvm::enumerate(Fields)) {
      auto &FieldQT = Field.value();

      const model::Type *FieldUnqualType = nullptr;

      if (FieldQT.UnqualifiedType().isValid())
        FieldUnqualType = FieldQT.UnqualifiedType().getConst();

      // Don't add edges for primitive types, as they would pollute the graph
      // and add no information regarding the type system structure
      if (not FieldUnqualType
          or FieldUnqualType->Kind() == model::TypeKind::PrimitiveType)
        continue;

      uint64_t SuccID;
      auto It = NodesMap.find(FieldUnqualType);
      if (It != NodesMap.end()) {
        // If a node already exists for the target type, use that
        SuccID = It->second;
      } else {
        // If the node does not already exist, create a new one
        SuccID = NextID;
        EmitNode(FieldUnqualType);
      }

      // Add an edge to the type referenced by the current field.
      // Since we have created the target type if it does not exist, and
      // the source node was either the root node or a successor of a
      // previously visited node, we are sure that both the source and the
      // destination of this edge have already been created.
      addFieldEdge(FieldQT, CurID, Field.index(), SuccID);

      // Push the field's type to the visit stack
      ToVisit.push_back(FieldUnqualType);
    }

    // Mark this Type as visited: the node has been emitted, as well as
    // all of its outgoing edges and their respective target nodes.
    Visited.insert(CurType);
  }
}

void TypeSystemPrinter::dumpFunctionNode(const model::Function &F, int NodeID) {
  const model::Type *PrototypeT = F.Prototype().getConst();

  // Print the name of the node
  Out << "node_" << to_string(NodeID) << "[";

  // Choose the node's border color
  auto Color = Purple;
  Out << "color=" << Color << ", ";

  // Start of HTML-style label
  Out << "label= < <TABLE " << TableOpts << ">";

  // Print the name of the function on top
  Out << "<TR><TD bgcolor=" << Color << " " << PaddingOpts << "><B>" << F.name()
      << "()</B></TD></TR>";

  // Print connected types in a table
  Out << "<TR><TD><TABLE " << TableOpts << "> ";
  // Header
  Out << "<TR>";
  headerCell(Out, Color, "Prototype");
  headerCell(Out, Color, "StackType");
  Out << "</TR>";

  // Second row of the inner table (actual types)
  Out << "<TR>";
  paddedCell(Out, PrototypeT->name(), /*port=*/0);
  if (F.StackFrameType().isValid() and not F.StackFrameType().empty()) {
    const model::Type *StackT = F.StackFrameType().getConst();
    paddedCell(Out, StackT->name(), /*port=*/1);
  } else {
    Out << "<TD></TD>";
  }
  Out << "</TR>";

  // End of inner table
  Out << "</TABLE></TD></TR>";

  // End of label
  Out << "</TABLE> >];\n";
}

void TypeSystemPrinter::print(const model::Function &F) {
  // Node corresponding to the function
  auto FunctionNodeID = NextID;
  dumpFunctionNode(F, FunctionNodeID);
  NextID++;

  // Nodes of the subtypes if they do not already exist
  const model::Type *PrototypeT = F.Prototype().getConst();
  bool HasStackFrame = F.StackFrameType().isValid()
                       and not F.StackFrameType().empty();
  const model::Type *StackT = HasStackFrame ? F.StackFrameType().getConst() :
                                              nullptr;

  print(*PrototypeT);
  if (StackT)
    print(*StackT);

  // Edges
  auto PrototypeNodeID = NodesMap.at(PrototypeT);
  addEdge(FunctionNodeID, 0, PrototypeNodeID);
  if (StackT) {
    auto StackNodeID = NodesMap.at(StackT);
    addEdge(FunctionNodeID, 1, StackNodeID);
  }
}

void TypeSystemPrinter::print(const model::Binary &Model) {
  // Print all functions and related types
  for (auto &F : Model.Functions())
    print(F);

  // Print remaining types, if any
  for (auto &T : Model.Types()) {
    if (NodesMap.contains(T.get()))
      continue;

    // Avoid polluting the graph with uninformative nodes
    if (T->Kind() != model::TypeKind::PrimitiveType and not T->edges().empty())
      print(*T);
  }
}
