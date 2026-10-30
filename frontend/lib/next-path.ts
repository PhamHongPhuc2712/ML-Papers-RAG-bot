/** Where to go after logging in: a same-site path only. An absolute URL, `//host` or `/\host` (which
 *  browsers read as `//host`) would be an open redirect, so they become "/". */
export function safeNext(next: string | null | undefined): string {
  return next && next.startsWith("/") && !next.startsWith("//") && !next.startsWith("/\\") ? next : "/";
}
