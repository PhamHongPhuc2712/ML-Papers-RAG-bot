import type { Metadata } from "next";
import { Inter } from "next/font/google";
import { cookies } from "next/headers";
import type { ReactNode } from "react";
import { Shell } from "@/components/Shell";
import { SESSION_COOKIE, corpus, currentUser } from "@/lib/server";
import { SHELL_INIT } from "@/lib/theme";
import "./globals.css";

// Downloaded at build time and served from this server: a reader's browser never calls Google.
const inter = Inter({ subsets: ["latin"], variable: "--font-inter", display: "swap" });

export const metadata: Metadata = {
  title: "ML Research Copilot",
  description: "Ask about ML papers: answers written from their text, with citations and References.",
};

export default async function RootLayout({ children }: { children: ReactNode }) {
  const signedIn = Boolean((await cookies()).get(SESSION_COOKIE)?.value);
  const [user, facts] = await Promise.all([signedIn ? currentUser() : null, corpus()]);
  return (
    <html lang="en" className={inter.variable} suppressHydrationWarning>
      <head>
        <script dangerouslySetInnerHTML={{ __html: SHELL_INIT }} />
      </head>
      <body>
        <Shell signedIn={signedIn} email={user?.email ?? null} papers={facts?.papers ?? null}>{children}</Shell>
      </body>
    </html>
  );
}
