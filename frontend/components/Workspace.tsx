import type { ReactNode } from "react";

/** A signed-in page's area beside the sidebar: the home page or a conversation with its panel. */
export function Workspace({ children }: { children: ReactNode }) {
  return <main className="workspace">{children}</main>;
}
