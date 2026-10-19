# The web app's redesign (UI plan, 2026-10-16): Implementation Plan

**Decision (the developer, 2026-10-16).** The one-page app works but "looks so bad". The product
should become a real product people sign up for and use. Order: **this UI redesign first**, then
sign-up, then deployment, each its own plan reviewed before it is built.

**The design is the sketch canvas**, reviewed by the developer:
[ML Research Copilot — page sketches](https://claude.ai/artifact/PbMka6GSH3o5BqhRtTcZhu) (private
to the developer). It follows **Consensus's layout and flows** (studied signed in, read-only, on
2026-10-16: sidebar, centred ask box with a toolbar, suggestion buttons, the question as a bubble,
a steps box, the References side panel that turns into a paper view, email-code sign-in), with
**our own name, mark, colours and wording**, and **Inter**, DeepSeek's web-chat font. Consensus's
logo, copy, branded features (Consensus Meter, Pro/Deep modes) and component styling are not
copied.

**Settings (the developer's answers, 2026-10-16):**
- Name: **ML Research Copilot**; mark: a citation bracket around a yellow highlighter stroke.
- Font: **Inter** everywhere (400, 500, 600, 700). No serif.
- Theme: **light and dark**, following the system setting, with a System / Light / Dark switch.
- Colours: kept from the first draft; light is white with a light grey sidebar (below).
- References: **a side panel** beside the answer; a citation opens its paper and passages there.
- Phone layouts: **not designed now**; pages must still not break at phone width (the sidebar
  becomes a drawer, the panel a full-width overlay), but no phone-specific design work.
- Order: UI, then sign-up, then deploy.

**Not in scope.** No backend or API change: the redesign uses today's endpoints and SSE frames
(`chat`, `status`, `references`, `evidence`, `final`, `error`). Anything a sketch shows that needs
the backend is listed under "Later plans" and left out of the build, not faked.

## Design

### Colour tokens (`:root`, redefined for dark)

| Token | Light | Dark | Use |
|---|---|---|---|
| `--canvas` | `#FFFFFF` | `#1A1C22` | page background |
| `--rail` | `#F6F7F9` | `#15171C` | sidebar, suggestion buttons |
| `--surface` | `#FFFFFF` | `#22252D` | ask box, panel, menus |
| `--line` | `#E3E6EB` | `#33373F` | borders, dividers |
| `--ink` | `#1B1E24` | `#E7E9EE` | text |
| `--ink-soft` | `#5D6472` | `#A0A6B2` | secondary text |
| `--accent` | `#3A55C8` | `#93A6FF` | links, focus, send button, citation chips |
| `--accent-soft` | `#EEF1FC` | `#2A3150` | question bubble, selected states |
| `--mark` | `#FFE89A` | `#5A4A12` | highlighter: the cited claim and passage |

The only warm colour is the highlighter, used only where evidence is marked. Every text pair
at least 4.5:1, checked by axe in both themes.

### Type

Inter, loaded with `next/font/google` (downloaded at build time, served from our own server: no
request to Google from a reader's browser). Scale (px): 13 meta, 14 controls, 15 UI, 16 ask box
and question, 17 answer body (line-height 1.65, measure at most 72 characters), 20 panel title,
26 page headings. Sentence case; no all-caps labels.

### The screens (canvas boards 1–11)

1. **Home, signed out.** Sidebar: mark, collapse, New chat, Home, a short pitch, Log in and Sign
   up at the bottom (Sign up hidden until the sign-up plan, D1). Page: mark and name, "What do you want to know about ML research?", the ask
   box (text, a "Venues and years" filter button, send), three suggestion buttons, the
   conferences covered. Sending, a suggestion, or the filter while signed out opens sign-in
   with the question kept.
2–3. **Log in or sign up** (email, then a 6-digit code). Built by the sign-up plan with its
   backend; until then the existing email-and-password login page is restyled to this layout.
4. **Home, signed in.** Sidebar: mark, collapse, New chat, History, chats grouped Today /
   Previous 7 days / Older, the account at the bottom. Page as in 1.
5. **An answer being written.** Header: chat title, References (disabled until the list
   arrives). The question as a right-aligned bubble; a steps box that fills in live
   ("Searched 85,729 papers for “…”, 20 ranked", "Reading the top 10 papers", "Writing the
   answer"); placeholder lines; the follow-up box pinned at the bottom with Stop.
6. **Answer with References open.** The steps box collapsed to one line ("Searched for “…” and
   read 10 of 20 papers"); claims with numbered citation chips; Copy under the answer; the panel
   on the right: References (20), venue and year filters, "Cited only", each paper with rank,
   title, venue and year, "Cited n" and "Read".
7. **A citation opened.** The chip and its claim take the highlighter; the panel shows the paper
   (title, venue, year, Paper page and PDF links), the cited passage with section and pages and
   its cited part marked, and "n more passages read from this paper".
8. **Paper page.** Venue line, title, authors, PDF and source links, Copy citation, an "Ask about
   this paper" box with the paper attached, the abstract, full-text status, related papers.
9. **History.** Search box over chat titles; chats grouped by day with their age and a delete
   action.
10. **Settings.** Account email, theme, delete account (exists today); usage, password and
   "delete all chats" wait for the sign-up plan.
11. **Messages and states.** Daily limit reached, not enough evidence, answer failed (Try again),
   **partial search** (new: an answer whose search degraded says so, from the warnings the
   turn already stores), logged out, service unavailable, no chats yet, page not found.

### Behaviour

- The sidebar collapses to a 60px rail, remembered per browser; under 900px it is a drawer and
  the page behind it is inert.
- The References panel is 400px beside the thread; under 1200px it overlays the thread from the
  right; Esc closes it. Selecting a chip opens the panel on that paper.
- Motion only answers an action (panel 180 ms, sidebar 150 ms); `prefers-reduced-motion` makes
  both instant.

## Tasks

Each task: lint, typecheck, unit tests, the browser tests against the scripted backend with axe
in both themes on every page touched, `next build`, and screenshots in light and dark at 1440px,
compared with the canvas board before the task is called done.

### Task D1: Tokens, font, theme and the app shell

**Done 2026-10-16.** The shell, theme switch, grouped chat list, account menu and drawer
(`Shell.tsx`, `ChatList.tsx`, `AccountMenu.tsx`, `ThemeToggle.tsx`, `Icon.tsx`, `lib/theme.ts`,
`lib/use-root-data.ts`, `lib/chat-groups.ts`), first committed in `1fe813c`, then brought to the
sketches. The History link joins the sidebar in D4, with the page it opens.
- [x] Inter instead of IBM Plex Sans and Source Serif 4; the colour tokens above.
- [x] The sidebar of boards 1 and 4: New chat as a round plus, the signed-out Home link, pitch and
  Log in at the bottom. **Sign up is hidden until the sign-up plan builds it** (accounts are
  created only by `users create` until then); board 1's Sign up button appears with that plan.
- [x] A chat's delete stays in the sidebar as an icon shown on hover or keyboard focus (always
  on touch screens), as built, beside the delete action on the History page.

### Task D2: Home and the ask box

**Files:** `app/page.tsx`, new `components/Composer.tsx`, `components/Suggestions.tsx`,
`app/login/page.tsx` (restyled to board 2's layout, keeping email and password).
**Done 2026-10-16.** `Composer.tsx`, `Suggestions.tsx`, `Hero.tsx`, `SignedOutHome.tsx`,
`lib/corpus.ts` (venues, years and the paper count from `GET /v1/corpus/coverage`, also shown in
the sidebar pitch); the login page in board 2's layout says the question is kept.
- [x] The ask box: grows to 8 lines, Enter sends, Shift+Enter a new line, the count shows only
  past 1,800 of 2,000 characters, the send button becomes Stop while a turn runs. The "Venues and
  years" button opens a small form whose choice is added to the question's text (the search has
  no filter parameter today; a real filter is a later backend change).
- [x] Boards 1 and 4; signed out, sending goes to log in with the question kept (`?q=`).

### Task D3: The conversation and the panel

**Files:** `components/ChatView.tsx` (split into `Thread.tsx` and the stream logic), `Answer.tsx`,
new `StepsBox.tsx`, new `ReferencesPanel.tsx` (replaces `References.tsx` and `EvidencePanel.tsx`),
`lib/references.ts` (extended: passages grouped by paper, cited parts).
**Done 2026-10-19.** `ChatView.tsx` (the thread, its header and the panel state), `Answer.tsx`,
`StepsBox.tsx`, `ReferencesPanel.tsx` (list and paper views; the paper view fetches authors and
the PDF link from `GET /v1/papers/{id}`), `lib/references.ts`; `References.tsx` and
`EvidencePanel.tsx` deleted. The whole cited passage is marked, since a stored passage does not
record which of its sentences a claim rests on.
- [x] Boards 5–7: the bubble, the steps box live and collapsed, the claims with chips, Copy,
  the panel's list and paper views, the highlighter on the selected claim and passage.
- [x] Stop while streaming; the stopped turn reads "Stopped" and can be asked again.
- [x] The partial-search note from the stored warnings (board 11).

### Task D4: History, the other pages, the states, and the record

**Files:** new `app/history/page.tsx`, `app/papers/[id]/page.tsx`, `app/account/page.tsx`,
`app/not-found.tsx`, the e2e specs, `reports/ui-redesign.md`.
- [ ] Board 9 (searching titles in the browser over `GET /v1/chats`), board 8, board 10's existing
  parts, board 11's states in both themes.
- [ ] Browser tests for every behaviour of today's app plus collapse, theme, Stop, the panel,
  History search and the partial-search note; axe with no serious or critical violation in both
  themes.
- [ ] One real-stack walk with screenshots (about $0.01), recorded in `reports/ui-redesign.md`.

## Gate

**G7 The redesign:** boards 1 and 4–11 built in light and dark and matching the canvas; every
behaviour of today's app kept and tested; no serious or critical axe violation; CI green; one
recorded real-stack walk.

## Later plans (shown on the canvas, not built here)

- **Sign-up plan:** email-code sign-in and sign-up (boards 2–3) with its backend: codes, their
  expiry and rate limits, an email-sending service; per-user daily limits and the "answers left
  today" line; settings' usage, "delete all chats"; answer feedback (Helpful / Not helpful).
  An email service is new infrastructure, which the spec's zero-budget, on-this-machine rule
  does not allow yet: the developer's decision to record first. "Continue with Google" is left
  out unless asked for.
- **Deployment plan:** hosting a real product (a server with a GPU or a CPU-only reranker, the
  85,729-paper index, PostgreSQL, a domain, TLS) changes the same rule.
- **Phone layouts**, designed when the developer asks.
