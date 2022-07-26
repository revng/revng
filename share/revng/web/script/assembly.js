//
// This file is distributed under the MIT License. See LICENSE.md for details.
//

var instructionLengthMap = new Map();
var commentIndicator = undefined;
const addComment = function(object, scope, child, needsIndent) {
  if (needsIndent) {
    var ind = document.createElement("span");
    ind.dataset.scope = "indentation";
    ind.style.width = instructionLengthMap.get(object) + "ch";
    ind.style.display = "inline-block";
    object.append(ind);
  }

  var comment = document.createElement("span");
  comment.dataset.scope = scope;
  comment.append(" " + commentIndicator + " ");
  comment.append(child);
  object.append(comment);

  object.insertAdjacentHTML('beforeEnd', "<br />");

  return true;
};

// Annotate the instructions.
for (instruction of document.querySelectorAll('[data-scope="asm.instruction"][data-html-exclusive]')) {
  instructionLengthMap.set(instruction, instruction.textContent.length);
  instruction.id = instruction.dataset.locationDefinition.substring(1);

  for (mnemonic of instruction.querySelectorAll('[data-token^="asm.mnemonic"]')) {
    link = document.createElement("a");
    link.href = "#" + instruction.id;
    mnemonic.appendChild(link).appendChild(link.previousSibling);
  }

  // Parse the metadata.
  var metadata = instruction.dataset.htmlExclusive.split('|');
  if (metadata.length != 8)
    throw new Error("Unsupported instruction metadata size");

  var type = metadata[0];
  if (type != instruction.dataset.scope)
    throw new Error("Unsupported instruction metadata type");
  
  if (commentIndicator == undefined)
    commentIndicator = metadata[2];
  else if (commentIndicator != metadata[2])
    throw new Error("Inconsistent comment indicator information");

  var rawBytes = metadata[1];
  var error = metadata[4];
  var comments = metadata.slice(5, 7);
  
  // var opcode = metadata[3];

  // Annotation configuration!
  const shouldAnnotateBytes = true && rawBytes.length != 0;
  const shouldAnnotateAddress = true;

  // Add the annotations.
  if (shouldAnnotateBytes || shouldAnnotateAddress) {
    var annotations = document.createElement("div");
    annotations.dataset.scope = "annotation";
    annotations.append(commentIndicator + ' ');

    if (shouldAnnotateBytes) {
      var bytes = document.createElement("span");
      bytes.append("[");
      bytes.append(rawBytes);
      bytes.append("]");
      annotations.append(bytes);
    }

    if (shouldAnnotateBytes || shouldAnnotateAddress)
      annotations.append(" at ");

    if (shouldAnnotateAddress) {
      const addresses = instruction.id.split('/');

      var link = document.createElement("a");
      link.href = "#" + instruction.id;
      link.append(addresses[addresses.length - 1]);
      annotations.append(link);
    }

    instruction.prepend(annotations);
  }

  // Add the trailing comments.
  var needsIndent = false;
  if (error.length != 0)
    needsIndent = addComment(instruction, "error", error, needsIndent);
  if (comments[0].length != 0)
    needsIndent = addComment(instruction, "comment", comments[0], needsIndent);
  if (comments[1].length != 0)
    needsIndent = addComment(instruction, "comment", comments[1], needsIndent);
}

const sanitizeAddress = function(address) {
  const forbidden = [ ' ', ':', '!', '#', '?', '<',
                      '>', '/', '{', '}', '[', ']' ];

  var result = "";
  for (character of address)
    if (forbidden.indexOf(character) != -1)
      result += '_';
    else
      result += character;

  return result;
}

const makeLink = function(location, text) {
  const elements = location.split('/');
  if (elements.length == 0)
    throw new Error("Link to empty locations are not supported.");

  console.log(elements[2])
  console.log(sanitizeAddress(elements[2]))

  switch (elements[1]) {
    case 'function':
    case 'basic-block':
    case 'instruction':
      var link = document.createElement("a");
      link.dataset.scope = "link"
      link.href = "./" + sanitizeAddress(elements[2]) + ".html#" + location.substring(1);
      link.append(text);
      return link;
    default:
      throw new Error("Link to an unsupported location.");
  }
}

const commentHelper = function(value, object, needsIndent) {
  return addComment(object, "comment", value, needsIndent);
}
const targetHelper = function(prefix, location, name, suffix, object, needsIndent) {
  var comment = document.createElement("span");
  comment.append(prefix + " ");
  comment.append(makeLink(location, name));
  comment.append(suffix);
  return addComment(object, "comment", comment, needsIndent);
}

// Add target information.
for (asmFunction of document.querySelectorAll('[data-scope="asm.function"]')) {
  for (instruction of asmFunction.querySelectorAll('[data-scope="asm.instruction"][data-location-references][data-html-exclusive]')) {
    var comments = instruction.querySelectorAll('[data-scope="comment"]');
    var errors = instruction.querySelectorAll('[data-scope="error"]');
    var ind = comments.length != 0 || errors.length != 0;

    var references = instruction.dataset.locationReferences.split(',');
    if (references.length == 1 && references[0] == '')
      references = [];

    const metadata = instruction.dataset.htmlExclusive.split('|');
    const targetData = metadata[7].split(',');
    
    if (targetData.length == 0) {
      // No targets.
    } else if (targetData.length == 1) {
      // Single target.
      const parsed = targetData[0].split(":");
      if (references.length != parsed.length - 1)
        throw new Error("Metadata is inconsistent with references.");

      switch (parsed[0]) {
        case 'call':
          if (parsed.length != 3)
            throw new Error("Normal calls require two locations.");
          ind = targetHelper("calls", references[1], parsed[2], ",", instruction, ind);
          ind = targetHelper("then goes to", references[0], parsed[1], ".", instruction, ind);
          break;
        case 'noreturn-call':
          if (parsed.length != 2)
            throw new Error("Noreturn calls require one location.");
          ind = targetHelper("calls", references[0], parsed[1], ",", instruction, ind);
          ind = commentHelper("and does not return.", instruction, ind);
          break;
        case 'indirect-call':
          if (parsed.length != 3)
            throw new Error("Indirect calls require one location.");
          ind = commentHelper("calls unknown location,", instruction, ind);
          ind = targetHelper("then goes to", references[0], parsed[1], ".", instruction, ind);
          break;
        case 'indirect-noreturn-call':
          if (parsed.length != 1)
            throw new Error("Indirect noreturn calls don't need locations.");
          ind = commentHelper("calls unknown location,", instruction, ind);
          ind = commentHelper("and does not return.", instruction, ind);
          break;
        case 'jump':
          if (parsed.length != 2)
            throw new Error("Jumps require one location.");
          if (parsed[1] != "the next instruction")
            ind = targetHelper("always goes to", references[0], parsed[1], ".", instruction, ind);
          break;
        case 'indirect-jump':
        case 'return':
          if (parsed.length != 1)
            throw new Error("Indirect jumps don't need locations.");
          // Does nothing: maybe we should write something after all???
          break;
        default:
          throw new Error("Unknown target type.");
      }
    } else {
      // Many targets.
      var nextInstructionTarget = undefined;
      var invalidTargetCount = 0;
      var properTargets = [];

      var referenceCounter = 0;
      for (const target of targetData) {
        const parsed = target.split(":");

        if (references.length - referenceCounter < parsed.length - 1)
          throw new Error("Metadata is inconsistent with references.");

        if (parsed[0] == "indirect-jump") {
          ++invalidTargetCount;
        } else if (parsed[0] == "jump") {
          if (parsed[1] == "the next instruction") {
            nextInstructionTarget = [references[referenceCounter++], parsed[1]];
          } else {
            const pair = [references[referenceCounter++], parsed[1]];
            properTargets.push(pair)
          }
        } else {
          throw new Error("Only jump targets can be repeated.");
        }
      }

      if (referenceCounter != references.length)
        throw new Error("The reference count is not consistent with metadata");

      if (properTargets.length == 1 && nextInstructionTarget != undefined) {
        ind = targetHelper("if taken, goes to", properTargets[0][0], properTargets[0][1], ",", instruction, ind);
        ind = targetHelper("otherwise, goes to", nextInstructionTarget[0], nextInstructionTarget[1], ",", instruction, ind);
      } else {
        ind = commentHelper("known targets include:", instruction, ind)
        if (nextInstructionTarget != undefined)
          ind = targetHelper("- ", nextInstructionTarget[0], nextInstructionTarget[1], "", instruction, ind);
        for (const [location, name] of properTargets)
          ind = targetHelper("- ", location, name, "", instruction, ind);
        if (invalidTargetCount != 0)
          ind = commentHelper("- and more.", instruction, ind);
      }
    }
  }
}
