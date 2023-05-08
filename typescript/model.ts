/**
 * This file is distributed under the MIT License. See LICENSE.md for details.
 */

import { safeParseInt, genGuid } from "./tuple_tree";

const metaAddressTypes = [
    "Invalid",
    "Generic32",
    "Generic64",
    "Code_x86",
    "Code_x86_64",
    "Code_mips",
    "Code_mipsel",
    "Code_arm",
    "Code_arm_thumb",
    "Code_aarch64",
    "Code_systemz",
] as const;
type MetaAddressType = (typeof metaAddressTypes)[number];

const metaAddressRegex = new RegExp(
    "^(?<Address>(0x[0-9a-fA-F]+)|)" +
        `:(?<Type>${metaAddressTypes.join("|")})` +
        "(:(?<Epoch>\\d+))?" +
        "(:(?<AddressSpace>\\d+))?$"
);

export type IMetaAddress = string;

export class MetaAddress {
    address?: string;
    type: MetaAddressType;
    epoch?: number;
    addressSpace?: number;

    constructor(metaAddress: string) {
        const match = metaAddress.match(metaAddressRegex);
        if (match !== null && match.groups !== undefined) {
            this.type = match.groups["Type"] as MetaAddressType;
            this.address = match.groups["Address"];
            this.epoch = safeParseInt(match.groups["Epoch"]);
            this.addressSpace = safeParseInt(match.groups["AddressSpace"]);
        } else {
            this.type = "Invalid";
        }
    }

    toString(): string {
        function cleanStr(str?: string | number, end = true): string {
            if (str !== undefined) {
                return end ? `:${str}` : `${str}`;
            } else {
                return "";
            }
        }
        return (
            `${cleanStr(this.address, false)}:${this.type}` +
            `${cleanStr(this.epoch)}${cleanStr(this.addressSpace)}`
        );
    }

    toJSON(): string {
        return this.toString();
    }
}

export function genPrimitiveTypeGuid(rawObject: IType): bigint {
    const realObject = rawObject as IPrimitiveType;
    const index = PrimitiveTypeKindValues.findIndex((e) => e === realObject.PrimitiveKind);
    return BigInt(BigInt(index << 8) | realObject.Size);
}

const genEnumTypeGuid = genGuid;
const genTypedefTypeGuid = genGuid;
const genStructTypeGuid = genGuid;
const genUnionTypeGuid = genGuid;
const genCABIFunctionTypeGuid = genGuid;
const genRawFunctionTypeGuid = genGuid;
