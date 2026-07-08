/*
 * This file is distributed under the MIT License. See LICENSE.md for details.
 */

export const ICONS: Record<string, string> = {
    dots: '<svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor"><circle cx="3" cy="8" r="1.4"/><circle cx="8" cy="8" r="1.4"/><circle cx="13" cy="8" r="1.4"/></svg>',
    search: '<svg width="14" height="14" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.6"><circle cx="7" cy="7" r="4.3"/><line x1="10.4" y1="10.4" x2="14" y2="14" stroke-linecap="round"/></svg>',
    chevron:
        '<svg width="12" height="12" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.7"><path d="M6 3l5 5-5 5" stroke-linecap="round" stroke-linejoin="round"/></svg>',
    back: '<svg width="13" height="13" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.7"><path d="M9 3L4 8l5 5M4.3 8H13" stroke-linecap="round" stroke-linejoin="round"/></svg>',
};

export function Icon({ name, class: cls }: { name: string; class?: string }) {
    return (
        <span class={cls ? `ic ${cls}` : "ic"} dangerouslySetInnerHTML={{ __html: ICONS[name] }} />
    );
}
