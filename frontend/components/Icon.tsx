import type { SVGProps } from "react";

// Inline icons, 24-unit grid, drawn with the current text colour. Decorative: every use sits
// beside a text label or inside a control that has an accessible name.
const PATHS = {
  plus: "M12 5v14M5 12h14",
  sidebar: "M4 5h16v14H4zM9.5 5v14",
  menu: "M4 7h16M4 12h16M4 17h16",
  close: "M6 6l12 12M18 6L6 18",
  trash: "M5 7h14M10 7V5h4v2M7 7l1 12h8l1-12",
  sun: "M12 4v2M12 18v2M4 12h2M18 12h2M6.3 6.3l1.4 1.4M16.3 16.3l1.4 1.4M6.3 17.7l1.4-1.4M16.3 7.7l1.4-1.4M12 9a3 3 0 1 0 0 6a3 3 0 1 0 0-6",
  moon: "M19 14.5A7.5 7.5 0 0 1 9.5 5a7.5 7.5 0 1 0 9.5 9.5z",
  monitor: "M4 5h16v11H4zM9 20h6M12 16v4",
  user: "M12 12a4 4 0 1 0 0-8a4 4 0 1 0 0 8M5 20c1.2-3.5 4-5 7-5s5.8 1.5 7 5",
  signout: "M10 5H5v14h5M14 8l4 4l-4 4M18 12H9",
  signin: "M14 5h5v14h-5M10 8l4 4l-4 4M14 12H5",
  chevrons: "M8 9l4-4l4 4M8 15l4 4l4-4",
  home: "M4 11l8-6l8 6v8h-5v-5h-6v5H4z",
  filter: "M4 7h16M7 12h10M10 17h4",
  arrowUp: "M12 19V5M5 12l7-7l7 7",
  stop: "M7 7h10v10H7z",
  check: "M5 12l5 5L19 7",
  back: "M15 6l-6 6l6 6",
  copy: "M9 9h10v10H9zM5 15V5h10",
  thumbUp: "M7 11v9H4v-9zM7 11l4-7c1.5 0 2.5 1 2 3l-1 3h6a2 2 0 0 1 2 2.3l-1.2 6A2 2 0 0 1 16.8 20H7",
  thumbDown: "M7 13V4H4v9zM7 13l4 7c1.5 0 2.5-1 2-3l-1-3h6a2 2 0 0 0 2-2.3l-1.2-6A2 2 0 0 0 16.8 4H7",
  books: "M5 4h4v16H5zM10 4h4v16h-4zM15.5 4.5l3.8 1l-3.9 15l-3.8-1",
  external: "M14 5h5v5M19 5l-8 8M18 14v5H5V6h5",
  history: "M4 12a8 8 0 1 0 2.3-5.6M4 4v4h4M12 8v4l3 2",
  search: "M11 5a6 6 0 1 0 0 12a6 6 0 1 0 0-12M20 20l-4.5-4.5",
} as const;

export type IconName = keyof typeof PATHS;

export function Icon({ name, size = 20, ...rest }: { name: IconName; size?: number } & SVGProps<SVGSVGElement>) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.75}
      strokeLinecap="round" strokeLinejoin="round" aria-hidden="true" focusable="false" {...rest}>
      <path d={PATHS[name]} />
    </svg>
  );
}

/** The product's mark: a citation bracket around a highlighter stroke. */
export function Mark({ size = 26 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 28 28" aria-hidden="true" focusable="false">
      <path d="M9 5H5v18h4M19 5h4v18h-4" fill="none" stroke="currentColor" strokeWidth={2.4} strokeLinecap="round" strokeLinejoin="round" />
      <rect x="9.5" y="11.5" width="9" height="5" rx="1.5" fill="#ffd84d" stroke="currentColor" strokeWidth={1.4} />
    </svg>
  );
}
