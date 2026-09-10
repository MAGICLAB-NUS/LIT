# Project website

GitHub Pages serves `main` and `/docs` at https://magiclab-nus.github.io/LIT/.
The repository and website are public. Pushing changes to `main` triggers a
Pages deployment.
The site is fully static; no build step is needed. Images, videos, and the paper
are included under `docs/static/`.

This repository preserves the history of https://github.com/jianmanlincjx/LIT.
To bring in future updates from a clean checkout:

```sh
# Run once per new clone (the migration checkout already has this remote).
git remote add upstream https://github.com/jianmanlincjx/LIT.git

git switch main
git pull --ff-only origin main
git fetch upstream
git merge upstream/main
# Resolve any conflicts, then review the site before publishing.
git push origin main
```

Keep these NUS-specific changes when merging upstream updates:

- Video sources use local files under `docs/static/video/`, without pinned
  jsDelivr sources, including the JavaScript-generated real-robot gallery.
- `og:image` uses the absolute NUS Pages URL.
- Website code links point to `MAGICLAB-NUS/LIT`.

Upstream merges are manual; this repository does not automatically synchronize.
