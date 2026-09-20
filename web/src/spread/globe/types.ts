/** Publisher-country attention: where outlets covering the event are based. */
export type CountryMarker = {
  code: string;
  name: string;
  lat: number;
  lon: number;
  /** Playhead second at which the first article from this country is observed. */
  t: number;
  /** Playhead seconds of every article, ascending; drives marker growth. */
  articleTimes: number[];
  firstSeen: string;
  onset: string | null;
  lagHours: number | null;
  articles: number;
  effectiveReports: number;
  attentionRatio: number | null;
};

/** Where the event itself happened (GDELT geography of the incident). */
export type Origin = {
  lat: number;
  lon: number;
  title: string;
  location: string;
  globalOnset: string | null;
};

export type Hover =
  | { kind: "country"; marker: CountryMarker; x: number; y: number }
  | { kind: "origin"; origin: Origin; x: number; y: number };
