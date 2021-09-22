#pragma once

#include <ostream>

#include "Binary.h"
#include "TupleTreeDiff.h"
#include "Type.h"

//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

template<typename T>
concept IsMetaAddress = std::is_same_v<T, MetaAddress>;

template<typename T>
concept ObjectWithProperties =
  (not HasScalarOrEnumTraits<T>) and(HasTupleSize<T>)
  and (not std::is_base_of_v<model::Type, T>);

template<typename T>
concept IsScalarType =
  Boolean<T> or Integral<T> or(std::is_same_v<std::string, T>)
  or (std::is_same_v<model::Identifier, T>) or HasScalarOrEnumTraits<T>;

template<typename T>
concept SequenceLike =
  (IsKeyedObjectContainer<T> or IsStdVector<T>) and!HasScalarOrEnumTraits<T>;

class EnumInfoCollector {
public:
  explicit EnumInfoCollector() : EnumValues(){};

  template<typename T>
  void enumCase(T &Val, const char *Str, const T ConstVal) {
    EnumValues.template emplace_back(Str);
  }

  std::vector<std::string> EnumValues;
};

template<typename SchemaRootType>
class SchemaDumper {
private:
  int IndentLevel;
  bool CurrentLineAlreadyIndented;
  std::map<std::string, std::stringstream> TypeNameToSchema;
  std::string EmitTo;

public:
  explicit SchemaDumper() :
    IndentLevel(1), CurrentLineAlreadyIndented(false), TypeNameToSchema() {
    writeMetaAddressSchema();
    writeTypeRefSchema();
    emitSchemaRecursively<SchemaRootType>();
  };

  void dumpSchema(std::ostream &OS) {
    auto RootTypeName = getClassName<SchemaRootType>();
    OS << R"("$schema": "http://json-schema.org/draft-07/schema#")"
       << "\n";
    OS << R"("$ref": "#/definitions/)" << RootTypeName << "\"\n";
    OS << R"("title": ")" << RootTypeName << "\"\n";
    OS << "definitions:\n";

    for (const auto &MapEntry : TypeNameToSchema) {
      const auto &TypeName = MapEntry.first;
      const auto &Schema = MapEntry.second;
      OS << Schema.str() << "\n";
    }
  }

private:
  class IndentGuard {
  private:
    SchemaDumper *SD;

  public:
    explicit IndentGuard(SchemaDumper *SD) : SD(SD) { SD->IndentLevel++; }

    ~IndentGuard() { SD->IndentLevel--; }
  };

  class EmitToGuard {
  private:
    SchemaDumper *SD;
    std::string PrevEmitTo;

  public:
    explicit EmitToGuard(SchemaDumper *SD, std::string EmitTo) : SD(SD) {
      PrevEmitTo = SD->EmitTo;
      SD->EmitTo = EmitTo;
      if (not SD->TypeNameToSchema.contains(EmitTo)) {
        SD->TypeNameToSchema[EmitTo] = std::stringstream{};
      }
    }

    ~EmitToGuard() { SD->EmitTo = PrevEmitTo; }
  };

  // Handles objects (represented as a tuple-likes in C++)
  template<typename T>
  requires(not HasScalarOrEnumTraits<T>)
    and HasTupleSize<T> void emitSchemaRecursively() {
    auto ClassName = getClassName<T>();

    // If the schema for the current type was not already emitted, emit it
    if (!TypeNameToSchema.contains(ClassName)) {
      writeObjectSchema<T>();
      // Dump the schema for the tuple's properties as well
      emitObjectPropertiesSchemas<T>();
    }
  }

  // Emits the schemas for members of the given tuple
  template<typename T, size_t CurrentIndex = 0>
  requires(not HasScalarOrEnumTraits<T>)
    and HasTupleSize<T> void emitObjectPropertiesSchemas() {
    if constexpr (CurrentIndex < std::tuple_size_v<T>) {
      using tuple_element = std::decay_t<std::tuple_element_t<CurrentIndex, T>>;
      emitSchemaRecursively<tuple_element>();
      emitObjectPropertiesSchemas<T, CurrentIndex + 1>();
    }
  }

  // Emits schema for KeyedObjectContainers and vectors inner types
  template<typename T>
  requires SequenceLike<T>
  void emitSchemaRecursively() {
    using value_type = std::decay_t<typename T::value_type>;
    emitSchemaRecursively<value_type>();
  }

  // Handle UpcastablePointers
  template<typename T>
  requires(not HasScalarOrEnumTraits<T>)
    and IsUpcastablePointer<T> void emitSchemaRecursively() {
    using concrete_types = typename T::concrete_types;
    concrete_types *DummyPtr = static_cast<concrete_types *>(nullptr);
    emitUpcastablePointerConcreteTypesSchemas<>(DummyPtr);
  }

  template<typename... T>
  void emitUpcastablePointerConcreteTypesSchemas(const std::tuple<T...> *) {
    (emitSchemaRecursively<T>(), ...);
  }

  // Handles enums
  template<typename T>
  requires HasScalarEnumTraits<T>
  void emitSchemaRecursively() {
    auto Name = getClassName<T>();
    if (!TypeNameToSchema.contains(Name)) {
      writeEnumSchema<T>();
    }
  }

  // Handles scalars (scalars are base types, so no schema is emitted)
  template<typename T>
  requires IsScalarType<T>
  void emitSchemaRecursively() {}

  // Emits schema for MetaAddress scalars
  void writeMetaAddressSchema() {
    EmitToGuard ETG(this, "MetaAddress");
    write("MetaAddress:");
    {
      IndentGuard G(this);
      write("type: string");
      write("pattern: ", false);
      writeQuoted( // Address
        R"(0x[0-9a-fA-F]+)"
        // Type
        // TODO: generate this from MetaAddressType at compile time
        ":(Invalid|Generic32|Generic64|Code_x86|Code_x86_64|Code_mips"
        "|Code_mipsel|Code_arm|Code_arm_thumb|Code_aarch64|Code_systemz)"
        // Optional epoch
        R"((:\\d+)?)"
        // Optional address space
        R"((:\\d+)?)");
    }
  }

  // Emits schema for TypeRef scalars
  void writeTypeRefSchema() {
    EmitToGuard ETG(this, "TypeRef");
    write("TypeRef:");
    {
      IndentGuard G(this);
      write("type: string");
      write(R"EOF(description: "Reference to a type )EOF"
            R"EOF((e.g. /Types/Primitive-1234)")EOF");
    }
  }

  // Emits the schema for an object (represented as a tuple-like in C++)
  template<typename T>
  requires(not HasScalarOrEnumTraits<T>)
    and HasTupleSize<T> void writeObjectSchema() {
    auto ClassName = getClassName<T>();

    EmitToGuard ETG(this, ClassName);
    write(ClassName + ":");
    {
      IndentGuard IG1(this);
      write("type: object");

      write("title: ", false);
      write(ClassName);

      write("additionalProperties: false");

      write("required: [", false);
      writeObjectRequiredPropertyNames<T>();
      write("]");

      write("properties:");
      {
        IndentGuard IG2(this);
        writeObjectProperties<T>();
      }
    }
  }

  // Writes the list of required properties for an object
  template<typename T, size_t CurrentIndex = 0, bool EmitComma = false>
  requires(not HasScalarOrEnumTraits<T>)
    and HasTupleSize<T> void writeObjectRequiredPropertyNames() {
    if constexpr (CurrentIndex < std::tuple_size_v<T>) {
      using FieldType = typename TupleLikeTraits<T>::Fields;
      constexpr auto Field = static_cast<FieldType>(CurrentIndex);
      auto FieldName = TupleLikeTraits<T>::FieldsName[CurrentIndex];

      if (not llvm::yaml::MappingTraits<T>::template isOptional<Field>()) {
        if constexpr (EmitComma) {
          write(", ", false);
        }
        writeQuoted(FieldName, false);
        writeObjectRequiredPropertyNames<T, CurrentIndex + 1, true>();
      } else {
        writeObjectRequiredPropertyNames<T, CurrentIndex + 1, EmitComma>();
      }
    }
  }

  // Writes the object properties (and their schema recursively)
  template<typename T, size_t CurrentIndex = 0>
  requires ObjectWithProperties<T>
  void writeObjectProperties() {
    if constexpr (CurrentIndex < std::tuple_size_v<T>) {
      using tuple_element = std::decay_t<std::tuple_element_t<CurrentIndex, T>>;
      constexpr auto FieldName = TupleLikeTraits<T>::FieldsName[CurrentIndex];
      write(FieldName, false);
      write(":");
      {
        IndentGuard G(this);
        writeScalarTypeOrReference<tuple_element>();
      }
      writeObjectProperties<T, CurrentIndex + 1>();
    }
  }

  // Writes the object properties (and their schema recursively)
  template<typename T, size_t CurrentIndex = 0>
  requires std::is_base_of_v<model::Type, T>
  void writeObjectProperties() {
    if constexpr (CurrentIndex < std::tuple_size_v<T>) {
      using tuple_element = std::decay_t<std::tuple_element_t<CurrentIndex, T>>;
      constexpr auto FieldName = TupleLikeTraits<T>::FieldsName[CurrentIndex];
      if (FieldName == "Kind") {
        write("Kind: ");
        {
          IndentGuard G(this);
          write("enum: [", false);
          writeQuoted(model::TypeKind::getName(T::AssociatedKind).str(), false);
          write("]");
          write("default: ", false);
          writeQuoted(model::TypeKind::getName(T::AssociatedKind).str());
        }
      } else {
        write(FieldName, false);
        write(":");
        {
          IndentGuard G(this);
          writeScalarTypeOrReference<tuple_element>();
        }
      }

      writeObjectProperties<T, CurrentIndex + 1>();
    }
  }

  template<typename T>
  requires HasScalarEnumTraits<T>
  void writeEnumSchema() {
    EnumInfoCollector EIC;
    T DummyEnumInstance;
    llvm::yaml::ScalarEnumerationTraits<T>::enumeration(EIC, DummyEnumInstance);
    {
      auto Name = getClassName<T>();
      EmitToGuard G(this, Name);
      write(Name, false);
      write(":");
      {
        IndentGuard IG(this);
        write("enum: [", false);
        for (const auto &EnumerateResult : llvm::enumerate(EIC.EnumValues)) {
          auto V = EnumerateResult.value();
          auto Idx = EnumerateResult.index();
          if (Idx > 0) {
            write(", ", false);
          }
          writeQuoted(V, false);
        }
        write("]");
      }
    }
  }

  // ---
  // --- Start "leaf" handlers
  // ---

  // Handle scalars
  template<typename T>
  requires HasScalarTraits<T>
  void writeScalarTypeOrReference() {
    if constexpr (IsMetaAddress<T>) {
      writeRefTo("MetaAddress");
    } else if constexpr (IsTupleTreeReference<T>) {
      writeRefTo("TypeRef");
    } else if constexpr (Boolean<T>) {
      write("type: boolean");
    } else if constexpr (Integral<T>) {
      write("type: integer");
    } else {
      write("type: string");
    }
  }

  // Handle tuple-likes
  template<typename T>
  requires HasTupleSize<T>
  void writeScalarTypeOrReference() {
    auto ClassName = getClassName<T>();
    writeRefTo(ClassName);
  }

  // Handle enums
  template<typename T>
  requires HasScalarEnumTraits<T>
  void writeScalarTypeOrReference() {
    auto Name = getClassName<T>();
    writeRefTo(Name);
  }

  // Handle KeyedObjectContainers and vectors
  template<typename T>
  requires IsKeyedObjectContainer<T> or IsStdVector<T>
  void writeScalarTypeOrReference() {
    write("type: array");
    write("items:");

    using value_type = std::decay_t<typename T::value_type>;
    {
      IndentGuard G(this);
      writeScalarTypeOrReference<value_type>();
    }
  }

  // Handle UpcastablePointers
  template<typename T>
  requires IsUpcastablePointer<T>
  void writeScalarTypeOrReference() {
    using concrete_types = typename T::concrete_types;
    write("anyOf:");
    {
      IndentGuard G(this);
      concrete_types *DummyPtr = static_cast<concrete_types *>(nullptr);
      writeUpcastablePointerAlternatives<>(DummyPtr);
    }
  }

  template<typename... T>
  void writeUpcastablePointerAlternatives(const std::tuple<T...> *) {
    (writeUpcastablePointerAlternative<T>(), ...);
  }

  template<typename T>
  void writeUpcastablePointerAlternative() {
    write("- ", false);
    {
      IndentGuard G(this);
      writeScalarTypeOrReference<T>();
    }
  }

  // ---
  // --- End "leaf" handlers
  // ---

  template<typename T>
  void write(T Something, bool nl = true) {
    std::stringstream &SS = TypeNameToSchema[EmitTo];
    if (not CurrentLineAlreadyIndented) {
      for (int i = 0; i < IndentLevel; i++) {
        SS << "  ";
      }
      CurrentLineAlreadyIndented = true;
    }

    SS << Something;
    if (nl) {
      SS << "\n";
      CurrentLineAlreadyIndented = false;
    }
  }

  template<typename T>
  void writeQuoted(T Something, bool nl = true) {
    write("\"", false);
    write(Something, false);
    write("\"", nl);
  }

  template<typename T>
  void writeRefTo(T Something) {
    write(R"("$ref": )", false);
    write(R"("#/definitions/)", false);
    write(Something, false);
    write(R"(")");
  }

  template<typename T>
  requires HasTupleLikeTraits<T>
  auto getClassName() {
    auto FullClassName = TupleLikeTraits<T>::Name;
    auto ClassNameLastPart = llvm::StringRef(FullClassName)
                               .rsplit("::")
                               .second.str();
    return ClassNameLastPart;
  }

  template<typename T>
  requires HasScalarEnumTraits<T>
  auto getClassName() { return llvm::yaml::ScalarEnumerationTraits<T>::Name; }
};
