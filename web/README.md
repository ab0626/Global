# Web

Two Vite entries share this package:

- `spread.html` — **Show the spread of …**: search a macro-event, then watch
  where its coverage appeared as red markers on a white globe (publisher
  country, chronological by GDELT observed time). Needs the attention store:

  ```sh
  GDELT_ATTENTION_DATA=data/store/<window> uv run uvicorn api.app:app --port 8000
  cd web && npm install && npm run dev   # open http://localhost:5173/spread.html
  ```

  It calls `/api/v2/attention/search`, `/countries`, `/events/{id}/spread`
  (`include_family=false`, `min_country_confidence=0.5`) and
  `/events/{id}/countries`. Markers sit at country centroids from
  `country_baseline.parquet`; countries without one are listed as "not drawn".
- `index.html` — the older GDELT Explorer described below.

# GDELT Explorer

The GDELT Explorer is a local web interface for searching and visualizing the
GDELT Events and Mentions slice served by the companion FastAPI backend.

## Start the interface

From the repository root, start the backend first:

```sh
GDELT_API_DATA=data/api uv run uvicorn api.app:app --reload --port 8000
```

In a second terminal, install the frontend dependencies and start Vite:

```sh
cd web
npm install
npm run dev
```

Open the local URL printed by Vite, usually
[`http://localhost:5173`](http://localhost:5173). The Vite development server
proxies every `/api` request to `http://127.0.0.1:8000`.

If the backend is running elsewhere, set `GDELT_API_URL` when starting Vite:

```sh
GDELT_API_URL=http://127.0.0.1:9000 npm run dev
```

For a production build:

```sh
npm run build
npm run preview
```

## Using the explorer

The dashboard updates all panels when the search or filters change:

- **Search bar** filters articles and linked event labels.
- **Sort** changes article ordering.
- **Quad-class chips** filter events by CAMEO cooperation/conflict class.
- **Volume timeline** shows matched article share over time. Click a point to
  zoom the time window; use **Reset** to return to the full dataset window.
- **Tone timeline and histogram** show the tone distribution of matched
  mentions.
- **Coverage map** shows linked `ActionGeo` locations. Select a map location
  to add it to the query.
- **Facets** show common domains, languages, themes, actors, and locations.
  Selecting a facet adds its query operator.
- **Events** lists linked CAMEO events. Select an event to open its article
  coverage drawer.
- **Articles** lists the matching documents and their derived metadata.
- **Data provenance** explains which fields come directly from the archive and
  which are derived or unavailable.

The initial query is `christchurch OR mosque`. Example queries:

```text
boeing OR 737 -flight
idai OR mozambique
brexit
theme:protest
location:"New Zealand"
sourcecountry:france
sourcelang:french
domainis:stuff.co.nz
quadclass:4
```

Supported query syntax includes bare terms, quoted phrases, `OR`, negation
with `-`, and the field operators `domain:`, `domainis:`, `sourcelang:`,
`sourcecountry:`, `theme:`, `location:`, `actor:`, and `quadclass:`.

## Data limitations

This interface uses the local Events and Mentions archive, not the complete
live GDELT service. The archive does not include article headlines, article
body text, social images, or mobile URLs:

- Article titles are derived from URL slugs.
- Search matches derived titles, domains, and linked CAMEO labels.
- `seendate` is the earliest GDELT observation time for an article.
- Publisher country is approximated from the domain country-code TLD.
- Social image and mobile URL fields are empty.

The backend's `/api/v2/ext/meta` endpoint exposes the full provenance map.
