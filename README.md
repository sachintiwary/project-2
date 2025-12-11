# LLM Quiz Solver

An automated quiz solver for TDS LLM Analysis Project. Receives quiz tasks via POST requests, renders JavaScript pages with Playwright, solves questions using GPT-4o, and submits answers.

## Features

- **Flask API** with secret verification
- **Playwright** for JavaScript page rendering
- **GPT-4o** via AI Pipe for question solving
- **Multi-format support**: Text, JSON, CSV, PDF, Audio, Images
- **Robust submission logic** - extracts submit URL from page content (no hallucination!)

## Quick Start

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt
playwright install chromium

# Set environment variables
export MY_SECRET="your-secret"
export AIPIPE_TOKEN="your-aipipe-token"

# Run
python app.py
```

### Deploy to Render

1. Push to GitHub
2. Connect repo to Render
3. Set environment variables:
   - `MY_SECRET`: Your secret string
   - `AIPIPE_TOKEN`: Your AI Pipe token
4. Deploy!

## API

### POST /solve

```json
{
  "email": "your-email@example.com",
  "secret": "your-secret",
  "url": "https://example.com/quiz-123"
}
```

**Responses:**
- `200`: Processing started
- `400`: Invalid JSON
- `403`: Invalid secret

## Environment Variables

| Variable | Description |
|----------|-------------|
| `MY_SECRET` | Your verification secret |
| `AIPIPE_TOKEN` | AI Pipe API token |
| `LLM_MODEL` | Model to use (default: gpt-4o) |

## License

MIT
