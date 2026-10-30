import { Mark } from "./Icon";

/** The home page's heading (canvas boards 1 and 4). */
export function Hero() {
  return (
    <div className="hero">
      <p className="hero-name"><Mark size={32} /> ML Research Copilot</p>
      <h1>What do you want to know about ML research?</h1>
    </div>
  );
}
