"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useRef, useState, type ReactNode } from "react";
import { setSidebarCollapsed } from "@/lib/theme";
import { useRootData } from "@/lib/use-root-data";
import { AccountMenu } from "./AccountMenu";
import { ChatList } from "./ChatList";
import { Icon, Mark } from "./Icon";
import { ThemeToggle } from "./ThemeToggle";

type Props = { signedIn: boolean; email: string | null; children: ReactNode };

/** Every page's frame: the sidebar (a rail when collapsed, a drawer on narrow screens) and the page. */
export function Shell({ signedIn, email, children }: Props) {
  const pathname = usePathname();
  const [drawer, setDrawer] = useState(false);
  const collapsed = useRootData("sidebar") === "collapsed";
  const opener = useRef<HTMLButtonElement>(null);
  const sidebar = useRef<HTMLElement>(null);
  const restoreFocus = useRef(false);

  // Following a link closes the drawer.
  const [path, setPath] = useState(pathname);
  if (path !== pathname) {
    setPath(pathname);
    setDrawer(false);
  }
  useEffect(() => {
    if (!drawer) {
      // Only once the page is no longer inert can the menu button take focus back.
      if (restoreFocus.current) opener.current?.focus();
      restoreFocus.current = false;
      return;
    }
    sidebar.current?.querySelector<HTMLElement>("a, button")?.focus();
    const escape = (event: KeyboardEvent) => {
      if (event.key === "Escape") closeDrawer();
    };
    document.addEventListener("keydown", escape);
    return () => document.removeEventListener("keydown", escape);
  }, [drawer]);

  function closeDrawer() {
    restoreFocus.current = true;
    setDrawer(false);
  }

  function toggleCollapsed() {
    setSidebarCollapsed(!collapsed);
  }

  return (
    <div className="app" data-drawer={drawer ? "open" : "closed"}>
      <aside className="sidebar" ref={sidebar} aria-label="Sidebar">
        <div className="sidebar-top">
          <nav aria-label="Main">
            <Link href="/" className="brand">
              <Mark />
              <span className="brand-name">ML Research Copilot</span>
            </Link>
          </nav>
          <button type="button" className="icon-button collapse-toggle" onClick={toggleCollapsed}
            aria-label={collapsed ? "Expand sidebar" : "Collapse sidebar"} aria-expanded={!collapsed}>
            <Icon name="sidebar" />
          </button>
          <button type="button" className="icon-button drawer-close" onClick={closeDrawer} aria-label="Close menu">
            <Icon name="close" />
          </button>
        </div>
        {signedIn ? (
          <>
            <Link href="/" className="side-link" onClick={() => setDrawer(false)}>
              <Icon name="plus" /> <span className="label">New chat</span>
            </Link>
            <ChatList onNavigate={() => setDrawer(false)} />
            <AccountMenu email={email} />
          </>
        ) : (
          <>
            <Link href={`/login?next=${encodeURIComponent(pathname)}`} className="side-link">
              <Icon name="signin" /> <span className="label">Sign in</span>
            </Link>
            <div className="sidebar-bottom label">
              <ThemeToggle />
            </div>
          </>
        )}
      </aside>
      <div className="scrim" onClick={closeDrawer} aria-hidden="true" />
      <div className="content" inert={drawer || undefined}>
        <div className="topbar">
          <button ref={opener} type="button" className="icon-button" onClick={() => setDrawer(true)}
            aria-label="Open menu" aria-expanded={drawer}>
            <Icon name="menu" />
          </button>
          <Link href="/" className="brand"><Mark size={22} /><span className="brand-name">ML Research Copilot</span></Link>
        </div>
        {children}
      </div>
    </div>
  );
}
