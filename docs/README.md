# Oscillation Adaptability Project Website

This directory contains the source files for the Oscillation Adaptability project website, hosted on GitHub Pages.

## Structure

- `index.html` - Main landing page
- `images/` - Directory containing all images used on the website
- `CNAME` - Custom domain configuration
- `sitemap.xml` - XML sitemap for search engines
- `robots.txt` - Instructions for web crawlers

## Development

To test the website locally, you can use any simple HTTP server. For example, with Python:

```bash
# Python 3
python -m http.server

# Python 2
python -m SimpleHTTPServer
```

Then visit `http://localhost:8000` in your browser.

## Deployment

The website is automatically deployed to GitHub Pages when changes are pushed to the main branch, using the GitHub Actions workflow defined in `.github/workflows/pages.yml`.

## Content Updates

When updating content:

1. Make sure all images are optimized for web (compressed, appropriate dimensions)
2. Update the `lastmod` date in `sitemap.xml` when making significant changes
3. Test all links to ensure they work correctly
4. Verify that the website looks good on both desktop and mobile devices
