# P7 Visualization Platform

A comprehensive web-based platform for testing, debugging, and visualizing grammars and constrained generation workflows using the P7 (Proposition 7) type-safe constrained generation engine.

## Features

### Grammar Editor
- Syntax-highlighted editor for `.spec` files
- Real-time grammar validation
- Error reporting with helpful suggestions
- Load from built-in examples (STLC, IMP, Fun)

### Grammar Debugger
- Test input parsing in real-time
- View valid completions at any point
- Check parse tree count and type information
- Visualize AST as S-expressions
- Identify type errors and grammar flaws

### Constrained Generation
- Side-by-side comparison of constrained vs unconstrained generation
- Integrated prompting interface
- Support for local models (GPT-2, Pythia, etc.)
- Real-time token streaming visualization
- Track generation completeness

## Architecture

```
api/
├── app.py              # Flask API server
└── requirements.txt    # Python dependencies

demo/
├── public/             # Static assets + runtime config
├── src/                # React components
│   ├── components/
│   │   ├── Header.js
│   │   ├── GrammarEditor.js
│   │   ├── DebugPanel.js
│   │   └── GenerationPanel.js
│   ├── App.js
│   ├── config.js
│   └── index.js
└── package.json        # Node dependencies

visualization/
└── start.sh            # Local launcher (API + demo)
```

## Prerequisites

1. **Python Environment** (3.9+)
   - P7 library built and installed: `cd python && pip install -e .`
   - PyTorch and Transformers: `pip install torch transformers`

2. **Node.js** (16+)
   - For running the React frontend

## Installation & Setup

### 1. Build and Install P7 Python Library

```bash
cd /path/to/p7/python
pip install -e .
pip install flask flask-cors transformers torch
```

### 2. Install Frontend Dependencies

```bash
cd demo
npm install
```

### 3. Start the Backend Server

```bash
cd api
python app.py
```

The backend will start on http://localhost:5001

### 4. Start the Frontend Development Server

In a new terminal:

```bash
cd demo
npm start
```

The demo will start on http://localhost:3000 and proxy API requests to the backend.

## Deployment Notes (Configurable API URL)

The demo can point to different API servers without rebuilding.

1. Edit `demo/public/config.js` and set `API_BASE_URL`:

```js
window.__P7_CONFIG__ = { API_BASE_URL: "https://api.example.com" };
```

2. Serve the demo build as static files. The runtime config is loaded before the app bootstraps.

Fallback order:
- `window.__P7_CONFIG__.API_BASE_URL`
- `REACT_APP_API_URL` (build-time)
- `http://localhost:5001`

## Usage

### 1. Load a Grammar Example
- Click "Select..." in the header to load built-in examples
- Available: STLC (Simply Typed Lambda Calculus), IMP (Imperative), Fun (Functional)

### 2. Edit Your Grammar
- Modify the grammar spec in the left editor
- Real-time validation shows if the grammar is valid
- Errors appear at the bottom of the editor with helpful hints

### 3. Debug the Grammar
- Click on "Grammar Debug" tab
- Type test input to see:
  - Current parse status
  - Well-typed tree count
  - Valid completions
  - Type errors (if any)
- Click "View AST" to see the parse tree

### 4. Generate with Constraints
- Enter a prompt in the bottom panel
- Set an initial seed (partial expression)
- Select a model (requires local download)
- Click "Generate" to see side-by-side comparison:
  - **Constrained**: Type-safe generation following the grammar
  - **Unconstrained**: Raw model output

## Grammar Spec Format

The `.spec` format has two parts separated by blank lines:

### 1. Grammar Productions
```
Name(rule_name) ::= Symbol₁[bind] Symbol₂ ... | Alt₂ | ...
```

- **Terminals**: `'λ'` (literal) or `/[a-z]+/` (regex)
- **Bindings**: `Identifier[x]` - attaches name for typing rules
- **Rule annotation**: `Lambda(lambda)` - links to typing rule
- **Epsilon**: `ε` for nullable productions
- **Alternatives**: `|`-separated

### 2. Typing Rules
```
premise₁, premise₂
------------------- (rule_name)
conclusion
```

**Premises:**
- `Γ ⊢ e : τ` - Ascription (e has type τ)
- `x ∈ Γ` - Membership
- `τ₁ = τ₂` - Equality
- `τ₁ ⊆ τ₂` - Subtype

**Conclusions:**
- `τ` - Bare type
- `Γ(x)` - Context lookup
- `Γ → Γ[x:τ] ⊢ σ` - Context transform

## Example: Simply Typed Lambda Calculus

```
Identifier ::= /[A-Za-z_][A-Za-z0-9_]*
Variable(var) ::= Identifier[x]
Type ::= /[A-Za-z0-9_]+/ | Type '->' Type
Lambda(lambda) ::= 'λ' Identifier[a] ':' Type[τ] '.' Expression[e]
Expression ::= Variable | Lambda | Expression Expression

x ∈ Γ
----- (var)
Γ(x)

Γ[a:τ] ⊢ e : ?B
--------------- (lambda)
τ → ?B

Γ ⊢ r : ?A → ?B, Γ ⊢ l : ?A
----------------------------- (app)
?B
```

## API Endpoints

### Health & Grammar Management
- `GET /api/health` - Health check
- `GET /api/grammars` - List available grammars
- `GET /api/grammars/<name>` - Get grammar spec
- `POST /api/validate-grammar` - Validate grammar spec

### Debugging
- `POST /api/debug/grammar` - Debug grammar at input state
- `POST /api/debug/completions` - Get raw completions for a prefix
- `POST /api/debug/token-filter` - Map completions to model vocab
- `GET /api/debug/server` - Server process details
- `GET /api/debug/generations` - Active generation tracker
- `POST /api/get-completions` - Get valid completions
- `POST /api/parse-to-ast` - Parse to S-expression AST

### Generation
- `POST /api/generate-constrained` - Stream constrained generation (SSE)
- `POST /api/generate-unconstrained` - Stream unconstrained generation (SSE)

#### Streaming API Model

All generation endpoints stream Server-Sent Events (SSE) with `data: <json>` frames.

`POST /api/generate-constrained`

Request JSON:
```json
{
  "spec": "<grammar spec>",
  "prompt": "<model prompt/context>",
  "initial": "<optional grammar prefix>",
  "model": "gpt2",
  "max_tokens": 50,
  "grammar_tokens": 50,
  "stop_on_complete": false,
  "mask_whitespace": true
}
```

SSE events:
- `status` → `{ "type": "status", "message": "..." }`
- `token` → `{ "type": "token", "text": "...", "full_text": "...", "step": 0 }`
- `done` → `{ "type": "done", "reason": "max_tokens|complete|no_valid|type_error|model_error", "is_complete": false }`
- `error` → `{ "type": "error", "message": "..." }`

`POST /api/generate-unconstrained`

Request JSON:
```json
{
  "prompt": "<model prompt/context>",
  "model": "gpt2",
  "max_tokens": 50,
  "top_k": 50,
  "temperature": 1.0
}
```

SSE events:
- `status` → `{ "type": "status", "message": "..." }`
- `token` → `{ "type": "token", "text": "...", "full_text": "...", "step": 0 }`
- `done` → `{ "type": "done", "reason": "max_tokens|model_error", "full_text": "..." }`
- `error` → `{ "type": "error", "message": "..." }`

## Troubleshooting

### "ModuleNotFoundError: No module named 'proposition_7'"
- Make sure you've installed the P7 library: `pip install -e .` from the python directory

### Model download is slow
- First run will download the model from HuggingFace
- Models are cached locally for subsequent runs

### Grammar validation fails
- Check that productions have `::=`
- Ensure typing rules are separated from grammar by blank lines
- Verify regex patterns are valid
- Check that rule names in productions match typing rules

## Development

### Backend Development
The Flask backend provides REST API endpoints for:
- Grammar validation using the P7 `Grammar` class
- Parsing and type-checking via `ConstrainedGenerator`
- Model inference using HuggingFace Transformers

### Demo Development
React components:
- **Header**: Grammar selection and status
- **GrammarEditor**: Monaco-based editor with syntax highlighting
- **DebugPanel**: Interactive grammar testing
- **GenerationPanel**: Side-by-side constrained/unconstrained generation

## License

MIT 
