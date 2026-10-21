"use client";

import { useEffect, useId, useRef, useState, type FormEvent, type KeyboardEvent, type ReactNode } from "react";
import { filterLabel, type AskFilter } from "@/lib/corpus";
import { Icon } from "./Icon";

export const LIMIT = 2000;
const COUNT_FROM = 1800;

type Props = {
  value: string;
  onChange: (value: string) => void;
  /** The question and the filter chosen for it; the caller writes the filter into the question. */
  onSubmit: (text: string, filter: AskFilter) => void;
  placeholder: string;
  label?: string;
  running?: boolean;
  onStop?: () => void;
  venues?: string[];
  years?: number[];
  autoFocus?: boolean;
  /** What the question is about, shown inside the box above the text (canvas board 8). */
  attachment?: ReactNode;
  /** A line under the box, such as "3 answers left today" (sign-up plan, A3). */
  note?: string | null;
};

/** The ask box (canvas boards 1 and 4): grows with its text, Enter sends, Stop while a turn runs. */
export function Composer({ value, onChange, onSubmit, placeholder, label = "Ask about ML papers", running = false,
  onStop, venues = [], years = [], autoFocus = false, attachment = null, note = null }: Props) {
  const [filter, setFilter] = useState<AskFilter>({ venue: null, year: null });
  const [filtering, setFiltering] = useState(false);
  const field = useRef<HTMLTextAreaElement>(null);
  const panel = useRef<HTMLDivElement>(null);
  const ids = useId();
  const chosen = filterLabel(filter);
  const tooLong = value.length > LIMIT;
  const ready = Boolean(value.trim()) && !tooLong && !running;

  useEffect(() => {
    if (autoFocus) field.current?.focus();
  }, [autoFocus]);

  useEffect(() => {
    if (!filtering) return;
    panel.current?.querySelector<HTMLElement>("select")?.focus();
    const close = (event: PointerEvent) => {
      if (!panel.current?.parentElement?.contains(event.target as Node)) setFiltering(false);
    };
    document.addEventListener("pointerdown", close);
    return () => document.removeEventListener("pointerdown", close);
  }, [filtering]);

  function submit(event?: FormEvent) {
    event?.preventDefault();
    if (ready) onSubmit(value.trim(), filter);
  }

  function onKey(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === "Enter" && !event.shiftKey && !event.nativeEvent.isComposing) {
      event.preventDefault();
      submit();
    }
  }

  return (
    <form className="composer" onSubmit={submit}>
      {attachment ? <p className="composer-attachment">{attachment}</p> : null}
      <label htmlFor={`${ids}-text`} className="visually-hidden">{label}</label>
      <textarea id={`${ids}-text`} ref={field} rows={2} value={value} placeholder={placeholder}
        onChange={(e) => onChange(e.target.value)} onKeyDown={onKey}
        aria-describedby={value.length > COUNT_FROM ? `${ids}-count` : undefined} />
      <div className="composer-bar">
        {venues.length || years.length ? (
          <div className="composer-filter">
            <button type="button" className="chip-button" aria-expanded={filtering} aria-controls={`${ids}-filter`}
              onClick={() => setFiltering((open) => !open)}>
              <Icon name="filter" size={16} /> {chosen ?? "Venues and years"}
            </button>
            {filtering ? (
              <div className="filter-panel" id={`${ids}-filter`} ref={panel}
                onKeyDown={(e) => { if (e.key === "Escape") setFiltering(false); }}>
                <label htmlFor={`${ids}-venue`}>Venue</label>
                <select id={`${ids}-venue`} value={filter.venue ?? ""}
                  onChange={(e) => setFilter({ ...filter, venue: e.target.value || null })}>
                  <option value="">Any venue</option>
                  {venues.map((venue) => <option key={venue} value={venue}>{venue}</option>)}
                </select>
                <label htmlFor={`${ids}-year`}>Year</label>
                <select id={`${ids}-year`} value={filter.year ?? ""}
                  onChange={(e) => setFilter({ ...filter, year: e.target.value ? Number(e.target.value) : null })}>
                  <option value="">Any year</option>
                  {years.map((year) => <option key={year} value={year}>{year}</option>)}
                </select>
                <div className="filter-actions">
                  <button type="button" onClick={() => setFilter({ venue: null, year: null })}>Clear</button>
                  <button type="button" className="primary" onClick={() => setFiltering(false)}>Done</button>
                </div>
              </div>
            ) : null}
          </div>
        ) : <span />}
        <span className="composer-end">
          {value.length > COUNT_FROM ? (
            <span id={`${ids}-count`} className={tooLong ? "count over" : "count"}>{value.length} / {LIMIT}</span>
          ) : null}
          {running ? (
            <button type="button" className="send stop" onClick={onStop} aria-label="Stop">
              <Icon name="stop" size={16} />
            </button>
          ) : (
            <button type="submit" className="send" disabled={!ready} aria-label="Send">
              <Icon name="arrowUp" size={18} />
            </button>
          )}
        </span>
      </div>
      {note ? <p className="composer-note" role="status">{note}</p> : null}
    </form>
  );
}
