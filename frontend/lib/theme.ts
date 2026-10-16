export type Theme = "system" | "light" | "dark";

export const THEME_KEY = "theme";
export const SIDEBAR_KEY = "sidebar";

/** Runs before first paint (an inline script in the root layout): no flash of the wrong theme
 *  or of an expanded sidebar the person had collapsed. Storage may be unavailable. */
export const SHELL_INIT = `(function(){try{var d=document.documentElement;var t=localStorage.getItem("${THEME_KEY}");if(t==="light"||t==="dark")d.dataset.theme=t;if(localStorage.getItem("${SIDEBAR_KEY}")==="collapsed")d.dataset.sidebar="collapsed";}catch(e){}})();`;

export function applyTheme(theme: Theme): void {
  const root = document.documentElement;
  if (theme === "system") delete root.dataset.theme;
  else root.dataset.theme = theme;
  try {
    if (theme === "system") localStorage.removeItem(THEME_KEY);
    else localStorage.setItem(THEME_KEY, theme);
  } catch {
    // Private windows may refuse storage; the choice still holds for this page.
  }
}

export function setSidebarCollapsed(collapsed: boolean): void {
  const root = document.documentElement;
  if (collapsed) root.dataset.sidebar = "collapsed";
  else delete root.dataset.sidebar;
  try {
    if (collapsed) localStorage.setItem(SIDEBAR_KEY, "collapsed");
    else localStorage.removeItem(SIDEBAR_KEY);
  } catch {
    // As above.
  }
}
