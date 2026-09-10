### Base CSS (always loaded)

This base stylesheet is always loaded before your page. It provides **no visual
component classes** — you have full creative control and write your own
page-scoped `<style>` for each slide. The base only guarantees:

- **Fixed page size**: `.page` is exactly `959.76pt x 540pt` (16:9), with
  `overflow:hidden` and `page-break-after:always`. Put all page content inside
  `<div class="page" id="...">...</div>`.
- **Font stack**: an Office-friendly font family is applied to `.page` and all
  its descendants, so text renders consistently in PowerPoint/print.
- **Tokens**: `--paper` (page background, default white) and `--ink` (default
  text color). These exist only so the pipeline can normalize the page
  background; you are free to override colors in your own `<style>`.

### What you add yourself

Everything else — layout (flex/grid), colors, gradients, cards, bands, badges,
timelines, shadows, borders, typography sizes — you author directly in a single
page-scoped `<style>` block inside the page. Scope every selector under the
page id (e.g. `#page-x .card { ... }`) so slides in the same document never
collide.
