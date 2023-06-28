/**
 * This file is distributed under the MIT License. See LICENSE.md for details.
 */

import * as yaml from "yaml";

export const yamlParseOptions = {
    strict: false,
    logLevel: "silent",
    intAsBigInt: true,
} as const;

export const yamlOutParseOptions = {
    keepUndefined: false,
    sortMapEntries: true,
} as const;

export const yamlToStringOptions = {
    doubleQuotedAsJSON: false,
    defaultKeyType: yaml.Scalar.PLAIN,
    defaultStringType: yaml.Scalar.QUOTE_DOUBLE,
    directives: true,
} as const;

export type IReference = string;

/**
 * Reference class
 * Represents a pointer to another element of the Tuple Tree, which can be fetched via 'resolve'
 */
export class Reference<T, M> {
    reference: string;

    constructor(ref: string) {
        this.reference = ref;
    }

    toJSON(): string {
        return this.reference;
    }

    equals(other: unknown): boolean {
        if (other instanceof Reference<T, M>) {
            return this.reference == other.reference;
        } else {
            return false;
        }
    }

    resolve(root: M): T {
        return _getElementByPath<T>(this.reference, root)!;
    }

    isValid(): boolean {
        return this.reference !== "";
    }

    static parse<T, M>(ref?: string): Reference<T, M> {
        return new Reference(ref || "");
    }
}

/**
 * Generate a random hexadecimal string
 * @param length output string length
 * @returns a random string
 */
function randomBytes(length: number): string {
    let value = "";
    for (let i = 0; i < length; i++) {
        value += (Math.random() * 16).toString(16);
    }
    return value;
}

/**
 * Generate a random bigint, between [start, end)
 * @param start lower bound
 * @param end upper bound
 * @returns random bigint
 */
function randomBigint(start: bigint, end: bigint): bigint {
    const diff = end - start;
    let value = end;
    do {
        const randomLength = diff.toString().length * Math.random();
        value = BigInt(`0x${randomBytes(randomLength)}`);
    } while (value > diff);
    return value + start;
}

/**
 * Utility function that returns a bigint suitable for IDs
 * @returns guid ID
 */
export function genGuid(): bigint {
    return randomBigint(2n ** 10n + 1n, 2n ** 64n - 1n);
}

// Tree traversal

function getElementByPathArray<T>(path: string[], obj: any): T | undefined {
    const component = path[0];
    if (obj instanceof Array) {
        for (const elem of obj) {
            if (keyableObject(elem) && elem.key() == component) {
                if (path.length == 1) {
                    return elem as unknown as T;
                } else {
                    return getElementByPathArray(path.slice(1), elem);
                }
            }
        }
        return undefined;
    } else {
        if (component in obj) {
            if (path.length == 1) {
                return obj[component];
            } else {
                return getElementByPathArray(path.slice(1), obj[component]);
            }
        } else {
            return undefined;
        }
    }
}

function setElementByPathArray<T>(path: string[], obj: any, value: T): boolean {
    const component = path[0];
    if (obj instanceof Array) {
        const index = obj.findIndex((elem) => keyableObject(elem) && elem.key() === component);

        if (index >= 0) {
            if (path.length === 1) {
                obj[index] = value;
                return true;
            } else {
                return setElementByPathArray(path.slice(1), obj[index], value);
            }
        } else if (
            index === -1 &&
            path.length === 1 &&
            keyableObject(value) &&
            value.key() === component
        ) {
            // We're appending a value to an array
            obj.push(value);
            return true;
        } else {
            return false;
        }
    } else {
        if (component in obj) {
            if (path.length == 1) {
                obj[component] = value;
                return true;
            } else {
                return setElementByPathArray(path.slice(1), obj[component], value);
            }
        }
        return false;
    }
}

/**
 * Given a Tuple Tree path and a root object, resolve the object or return
 * undefined if the path is not present in the tuple tree
 */
export function _getElementByPath<T>(path: string, obj: any): T | undefined {
    return getElementByPathArray(path.split("/").slice(1), obj);
}

/**
 * Given a Tuple Tree path, a root object and a value, set the path with the
 * value provided and return true. If the path does not resolve, this returns
 * false
 */
export function _setElementByPath<T>(path: string, obj: any, value: T): boolean {
    return setElementByPathArray(path.split("/").slice(1), obj, value);
}

// Tree Diffing

interface IDiffSet {
    Changes: IDiff[];
}

export class DiffSet {
    Changes: Diff[];

    constructor(Changes: Diff[]) {
        this.Changes = Changes;
    }

    static parse(rawObject: IDiffSet): DiffSet {
        return new DiffSet(rawObject.Changes.map((e) => new Diff(e.Path, e.Add, e.Remove)));
    }

    toJSON(): IDiffSet {
        return {
            Changes: this.Changes.filter((e) => e.isValid()).map((e) => e.toJSON()),
        };
    }

    reversed(): DiffSet {
        return new DiffSet(this.Changes.map((e) => new Diff(e.Path, e.Remove, e.Add)));
    }

    static deserialize(input: string): DiffSet {
        return DiffSet.parse(yaml.parse(input));
    }

    serialize(): string {
        const doc = new yaml.Document(this, yamlOutParseOptions);
        return doc.toString(yamlToStringOptions);
    }
}

interface IDiff {
    Path: string;
    Add?: any;
    Remove?: any;
}

class Diff {
    Path: string;
    Add: any;
    Remove: any;

    constructor(Path: string, Add: any, Remove: any) {
        this.Path = Path;
        this.Add = Add;
        this.Remove = Remove;
    }

    toJSON(): IDiff {
        let ret = { Path: this.Path };
        if (this.Add !== undefined) {
            ret["Add"] = this.Add;
        }
        if (this.Remove !== undefined) {
            ret["Remove"] = this.Remove;
        }
        return ret;
    }

    isValid(): boolean {
        return this.Add !== undefined || this.Remove !== undefined;
    }
}

/**
 * Comm object
 * Represents a difference between 2 sets, where 'common' are the elements present in both sets
 * and only<n> the ones only present in set<n>
 */
interface Comm<T> {
    only1: Set<T>;
    only2: Set<T>;
    common: Set<T>;
}

function isNull(obj: any): obj is undefined | null {
    return obj === undefined || obj === null;
}

/**
 * Computes the comm of the object keys, pruning undefined and null keys
 * @param obj1 first object
 * @param obj2 second object
 * @returns Comm object with only1 containing non-null keys unique to obj1, only2 keys unique to
 *          obj2 and common with keys common to obj1 and obj2
 */
function objComm(obj1: any, obj2: any): Comm<string> {
    const only1 = new Set<string>();
    const only2 = new Set<string>();
    const common = new Set<string>();
    for (const element of new Set([...Object.keys(obj1), ...Object.keys(obj2)])) {
        if (element in obj1 && element in obj2) {
            if (!isNull(obj1[element]) && !isNull(obj2[element])) {
                common.add(element);
            } else if (!isNull(obj1[element]) && isNull(obj2[element])) {
                only1.add(element);
            } else if (!isNull(obj2[element]) && isNull(obj1[element])) {
                only2.add(element);
            }
        } else if (element in obj1 && !isNull(obj1[element])) {
            only1.add(element);
        } else if (element in obj2 && !isNull(obj2[element])) {
            only2.add(element);
        }
    }

    return { only1: only1, only2: only2, common: common };
}

/**
 * Computes the comm of 2 sets
 * @param set1 first set
 * @param set2 second set
 * @returns Comm object where only1 are the elements unique to set1, only2 are elements unique to
 *          set2 and common are elements in common between the two sets
 */
function setComm<T>(set1: Set<T>, set2: Set<T>): Comm<T> {
    const only1 = new Set<T>();
    const only2 = new Set<T>();
    const common = new Set<T>();

    for (const element of new Set([...set1, ...set2])) {
        if (set1.has(element) && set2.has(element)) {
            common.add(element);
        } else if (set1.has(element)) {
            only1.add(element);
        } else {
            only2.add(element);
        }
    }

    return { only1: only1, only2: only2, common: common };
}

function hasEquals(obj: any): obj is { equals: (other: unknown) => boolean } {
    return typeof obj === "object" && typeof obj?.equals === "function";
}

function isNativeType(obj: any): obj is string | bigint | boolean {
    return ["string", "bigint", "boolean", "number"].includes(typeof obj);
}

function keyableObject(obj: any): obj is { key: () => string } {
    return (
        obj !== null &&
        typeof obj === "object" &&
        "key" in obj &&
        typeof obj.key === "function" &&
        typeof obj.key() === "string"
    );
}

function keyableArray(obj: Array<any>): obj is { key: () => string }[] {
    return obj.map((e) => keyableObject(e)).every((e) => e);
}

function keyabaleArrayToMap(array: { key: () => string }[]): Map<string, any> {
    const ret = new Map<string, any>();
    for (const element of array) {
        ret.set(element.key(), element);
    }
    return ret;
}

export function _makeDiff<T>(tuple_tree_old: T, tuple_tree_new: T): DiffSet {
    return new DiffSet(makeDiffSubtree(tuple_tree_old, tuple_tree_new, ""));
}

export function makeDiffSubtree(obj_old: any, obj_new: any, prefix: string): Diff[] {
    const result: Diff[] = [];
    if (typeof obj_old != typeof obj_new) {
        return [];
    }
    if (obj_old instanceof Array) {
        if (keyableArray(obj_old) && keyableArray(obj_new)) {
            const map_old = keyabaleArrayToMap(obj_old);
            const map_new = keyabaleArrayToMap(obj_new);

            const map_comm = setComm(new Set(map_old.keys()), new Set(map_new.keys()));

            for (const element of map_comm.only1) {
                result.push(new Diff(`${prefix}`, undefined, map_old.get(element)));
            }
            for (const element of map_comm.only2) {
                result.push(new Diff(`${prefix}`, map_new.get(element), undefined));
            }
            for (const element of map_comm.common) {
                result.push(
                    ...makeDiffSubtree(
                        map_old.get(element),
                        map_new.get(element),
                        `${prefix}/${element}`
                    )
                );
            }
        } else {
            const array_old: Array<any> = [...obj_old];
            const array_new: Array<any> = [...obj_new];
            array_old.sort();
            array_new.sort();
            while (array_old.length > 0) {
                const head = array_old.shift();
                const new_idx = array_new.indexOf(head);
                if (new_idx !== -1) {
                    array_new.splice(new_idx, 1);
                } else {
                    result.push(new Diff(prefix, undefined, head));
                }
            }
            result.push(...array_new.map((e) => new Diff(prefix, e, undefined)));
        }
    } else {
        const obj_comm = objComm(obj_old, obj_new);
        for (const element of obj_comm.only1) {
            result.push(new Diff(`${prefix}/${element}`, "", obj_old[element]));
        }
        for (const element of obj_comm.only2) {
            result.push(new Diff(`${prefix}/${element}`, obj_new[element], ""));
        }
        for (const element of obj_comm.common) {
            if (isNativeType(obj_old[element]) && isNativeType(obj_new[element])) {
                if (obj_old[element] !== obj_new[element]) {
                    result.push(
                        new Diff(
                            `${prefix}/${element}`,
                            obj_new[element].toString(),
                            obj_old[element].toString()
                        )
                    );
                }
            } else if (hasEquals(obj_old[element]) && hasEquals(obj_new[element])) {
                if (!obj_old[element].equals(obj_new[element])) {
                    result.push(
                        new Diff(
                            `${prefix}/${element}`,
                            obj_new[element].toJSON().toString(),
                            obj_old[element].toJSON().toString()
                        )
                    );
                }
            } else {
                result.push(
                    ...makeDiffSubtree(obj_old[element], obj_new[element], `${prefix}/${element}`)
                );
            }
        }
    }
    return result;
}

function constructType<T>(typeInfo: TypeInfo, data: any): T {
    switch (typeInfo.ctor) {
        case "class":
            return new typeInfo.type(data);
        case "parse":
            return typeInfo.type.parse(data);
        case "enum":
        case "native":
            return typeInfo.type(data);
    }
}

type getTypeInfoType = (path: string | string[], root?: any) => TypeInfo;
export function _getTypeInfo(
    path: string | string[],
    root: any,
    TYPE_HINTS: Map<TupleTreeType, { [key: string]: TypeInfo }>
): TypeInfo | undefined {
    if (typeof path === "string") {
        path = path.split("/").slice(1);
    }
    const component = path[0];
    const root_hint = TYPE_HINTS.get(root);
    if (root_hint !== undefined && component in root_hint) {
        const type_info = root_hint[component];
        if (path.length === 1) {
            return type_info;
        } else {
            if (type_info.isArray && type_info.ctor === "parse" && "parseKey" in type_info.type) {
                const key = path[1];
                const key_parsed = type_info.type.parseKey(key);
                if ("Kind" in key_parsed) {
                    const specialized_type = Array.from(TYPE_HINTS.keys()).find(
                        (e) => "name" in e && e["name"] === key_parsed.Kind
                    );
                    return _getTypeInfo(path.slice(2), specialized_type, TYPE_HINTS);
                }
                return _getTypeInfo(path.slice(2), type_info.type, TYPE_HINTS);
            } else {
                return _getTypeInfo(
                    path.slice(type_info.isArray ? 2 : 1),
                    type_info.type,
                    TYPE_HINTS
                );
            }
        }
    }
    return undefined;
}

function getParentElementByPath<T>(path: string, obj: any): T | undefined {
    const parentPath = path.split("/").slice(1, -1);
    if (parentPath.length > 1) {
        return _getElementByPath(`/${parentPath.join("/")}`, obj);
    } else {
        return obj;
    }
}

export function _validateDiff<T>(obj: T, diffs: DiffSet, getTypeInfo: getTypeInfoType): boolean {
    for (const diff of diffs.Changes) {
        const target = _getElementByPath(diff.Path, obj);
        const info = getTypeInfo(diff.Path);
        if (info.isArray) {
            if (diff.Remove !== undefined) {
                if (target instanceof Array) {
                    const array_target = target.find(
                        (e) => yaml.stringify(e) === yaml.stringify(diff.Remove)
                    );
                    if (array_target === undefined) {
                        return false;
                    }
                } else {
                    return false;
                }
            }
            if (diff.Add !== undefined && target instanceof Array) {
                const array_target = target.find(
                    (e) => yaml.stringify(e) === yaml.stringify(diff.Add)
                );
                if (array_target !== undefined) {
                    return false;
                }
            }
        } else {
            if (
                diff.Remove !== undefined &&
                yaml.stringify(target) !== yaml.stringify(diff.Remove)
            ) {
                return false;
            }
            if (diff.Add !== undefined && diff.Remove === undefined && !isNull(target)) {
                return false;
            }
        }
    }
    return true;
}

export function _applyDiff<T>(
    obj: T,
    diffs: DiffSet,
    validateDiff: (obj: T, diffs: DiffSet) => boolean,
    getTypeInfo: getTypeInfoType,
    clone: (obj: T) => T
): [false] | [true, T] {
    if (!validateDiff(obj, diffs)) {
        return [false];
    }
    const new_obj = clone(obj);
    for (const diff of diffs.Changes) {
        let target = _getElementByPath(diff.Path, new_obj);
        const info = getTypeInfo(diff.Path);
        if (target instanceof Array) {
            if (diff.Remove !== undefined) {
                const idx = target.findIndex(
                    (e) => yaml.stringify(e) === yaml.stringify(diff.Remove)
                );
                target.splice(idx, 1);
            }

            if (diff.Add !== undefined) {
                if (target === undefined) {
                    const parentPath = diff.Path.split("/").slice(1);
                    const element = parentPath.pop();
                    const parent = _getElementByPath(parentPath.join("/"), new_obj)!;
                    parent[element!] = [];
                    target = parent[element!];
                }
                (target as any).push(constructType(info, diff.Add));
            }
        } else {
            const parent = getParentElementByPath(diff.Path, new_obj)!;
            const element = diff.Path.split("/").reverse()[0];
            true;
            if (diff.Remove !== undefined) {
                parent[element] = undefined;
            }
            if (diff.Add !== undefined && diff.Add !== "") {
                parent[element] = constructType(info, diff.Add);
            }
        }
    }

    return [true, new_obj];
}

// Type Hinting

type Constructor = new (rawObject: any) => any;
type SimpleParsable = { parse: (rawObject: any) => any };
type Parsable = SimpleParsable | (SimpleParsable & { parseKey: (key: string) => any });
type NativeParsable = (rawObject: any) => any;

export type TupleTreeType = Constructor | Parsable;

interface CommonTypeInfo {
    optional: boolean;
    isArray: boolean;
}

interface ConstructorType extends CommonTypeInfo {
    type: Constructor;
    ctor: "class";
}

interface ParsableType extends CommonTypeInfo {
    type: Parsable;
    ctor: "parse";
}

interface NativeType extends CommonTypeInfo {
    type: NativeParsable;
    ctor: "native";
}

interface EnumType extends CommonTypeInfo {
    type: NativeParsable;
    ctor: "enum";
    possibleValues: readonly string[];
}

export type TypeInfo = ConstructorType | ParsableType | NativeType | EnumType;

export function BigIntBuilder(rawObject: any): bigint {
    return BigInt(rawObject);
}
