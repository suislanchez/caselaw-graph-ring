# LegalGPT Website - Vercel Deployment

## Quick Deploy to Vercel

### Option 1: Vercel CLI

```bash
# Install Vercel CLI
npm i -g vercel

# Navigate to website directory
cd website

# Deploy
vercel

# For production
vercel --prod
```

### Option 2: GitHub Integration

1. Push to GitHub
2. Go to [vercel.com](https://vercel.com)
3. Import your repository
4. Set root directory to `website`
5. Deploy

## Project Structure for Vercel

```
website/
├── api/
│   └── index.py          # FastAPI serverless function
├── static/
│   ├── css/styles.css
│   └── js/sse-client.js
├── templates/            # Jinja2 templates
│   ├── base.html
│   ├── index.html
│   ├── methodology.html
│   ├── data.html
│   ├── results.html
│   ├── agents.html
│   └── demo.html
├── vercel.json           # Vercel configuration
└── requirements.txt      # Python dependencies
```

## Environment Variables (Optional)

Set these in Vercel dashboard if needed:

- `GRADIO_URL` - URL to your HuggingFace Space for the demo

## Limitations on Vercel

1. **No SSE/WebSocket** - Agent dashboard uses polling instead (every 2s)
2. **10s timeout** - Serverless functions timeout after 10 seconds
3. **Read-only filesystem** - Can't write status files in production

## For Real-Time Updates

For true real-time agent status, consider:

1. **Vercel + External API**: Deploy website on Vercel, run API on Railway/Render
2. **Full self-hosted**: Use the local FastAPI app with `uvicorn`
3. **HuggingFace Spaces**: Deploy as Gradio app with full capabilities

## Local Development

```bash
# Run locally with hot reload
cd website
pip install -r requirements.txt
uvicorn api.index:app --reload --port 8000

# Or use the full app
cd ..
python -m uvicorn website.app:app --reload --port 8000
```

## Demo Integration

For the interactive demo, deploy a Gradio app to HuggingFace Spaces:

1. Create a Space at huggingface.co/spaces
2. Upload `src/demo/app.py`
3. Update `gradio_url` in `api/index.py` with your Space URL
