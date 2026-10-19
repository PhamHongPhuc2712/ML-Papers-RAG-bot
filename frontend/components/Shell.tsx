"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useRef, useState, type ReactNode } from "react";
import { setSidebarCollapsed } from "@/lib/theme";
import { useRootData } from "@/lib/use-root-data";
import { AccountMenu } from "./AccountMenu";
import { ChatList } from "./ChatList";
import { Icon, Mark } from "./Icon";

type Props = { signedIn: boolean; email: string | null; papers: number | null; children: ReactNode };

/** Every page's frame: the sidebar (a rail when collapsed, a drawer on narrow screens) and the page. */
export function Shell({ signedIn, email, papers, children }: Props) {
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
              {/* Signed out, the page itself shows the name beside the mark (canvas board 1). */}
              <span className={signedIn ? "brand-name" : "visually-hidden"}>ML Research Copilot</span>
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
        <Link href="/" className="side-link new-chat" onClick={() => setDrawer(false)}>
          <span className="round" aria-hidden="true"><Icon name="plus" size={16} /></span>
          <span className="label">New chat</span>
        </Link>
        {signedIn ? (
          <>
            <Link href="/history" className="side-link" aria-current={pathname === "/history" ? "page" : undefined}>
              <Icon name="history" size={18} /> <span className="label">History</span>
            </Link>
            <ChatList onNavigate={() => setDrawer(false)} />
            <AccountMenu email={email} />
          </>
        ) : (
          <>
            <Link href="/" className="side-link" aria-current={pathname === "/" ? "page" : undefined}>
              <Icon name="home" size={18} /> <span className="label">Home</span>
            </Link>
            <div className="pitch label">
              <p className="pitch-title">Answers from ML papers</p>
              <p>Ask anything about machine learning research.</p>
              <p>Every claim cites the <mark>passage it comes from</mark>, in {papers ? `${papers.toLocaleString("en")} ` : ""}conference papers.</p>
            </div>
            {/* Sign up appears with the sign-up plan; until then accounts are made by `users create`. */}
            <div className="sidebar-bottom signed-out">
              <Link href={`/login?next=${encodeURIComponent(pathname)}`} className="outline-button">
                <Icon name="signin" size={18} /> <span className="label">Log in</span>
              </Link>
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
