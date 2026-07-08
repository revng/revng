/*
 * This file is distributed under the MIT License. See LICENSE.md for details.
 */

const VARIANTS: [string, (end: string) => string][] = [
    ["bottomup", (end) => `Bottom up - ${end} location at the top`],
    ["topdown", () => "Top down - thread entry point at the bottom"],
];

export function Flamegraphs({
    prefix,
    filter,
    end,
}: {
    prefix: string;
    filter: string;
    end: string;
}) {
    return (
        <div class="fg-images">
            {VARIANTS.map(([type, caption]) => {
                const base = filter === "all" ? `${prefix}_${type}` : `${prefix}_${filter}_${type}`;
                const text = caption(end);
                return (
                    <a href={`${base}.svg`} target="_blank" title={text}>
                        <img src={`${base}.png`} alt={text} loading="lazy" />
                    </a>
                );
            })}
        </div>
    );
}
