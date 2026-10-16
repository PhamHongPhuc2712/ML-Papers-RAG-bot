import type { Metadata } from "next";
import { IBM_Plex_Sans, Source_Serif_4 } from "next/font/google";
import { cookies } from "next/headers";
import type { ReactNode } from "react";
import { Shell } from "@/components/Shell";
import { SESSION_COOKIE, currentUser } from "@/lib/server";
import { SHELL_INIT } from "@/lib/theme";
import "./globals.css";

// Downloaded at build time and served from this server: a reader's browser never calls Google.
const plex = IBM_Plex_Sans({ subsets: ["latin"], weight: ["400", "500", "600"], variable: "--font-plex", display: "swap" });
const serif = Source_Serif_4({ subsets: ["latin"], axes: ["opsz"], variable: "--font-serif", display: "swap" });

export const metadata: Metadata = {
  title: "ML Research Copilot",
  description: "Ask about ML papers: answers written from their text, with citations and References.",
};

export default async function RootLayout({ children }: { children: ReactNode }) {
  const signedIn = Boolean((await cookies()).get(SESSION_COOKIE)?.value);
  const user = signedIn ? await currentUser() : null;
  return (
    <html lang="en" className={`${plex.variable} ${serif.variable}`} suppressHydrationWarning>
      <head>
        <script dangerouslySetInnerHTML={{ __html: SHELL_INIT }} />
      </head>
      <body>
        <Shell signedIn={signedIn} email={user?.email ?? null}>{children}</Shell>
      </body>
    </html>
  );
}
