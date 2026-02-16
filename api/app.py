"""
P7 Visualization Platform - Flask Backend API
Provides endpoints for grammar validation, debugging, and constrained generation.
"""

from __future__ import annotations

import os
import sys
import traceback
import logging
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, List

# Disable tqdm progress bars to avoid BrokenPipeError in server environments
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

from flask import Flask, request, jsonify, Response
import threading
import time
from flask_cors import CORS

# Add parent directory to path to import p7
_parent = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_parent))

# Also add the python directory for editable installs
if (_parent / 'p7').exists():
    sys.path.insert(0, str(_parent))

import p7 as p7
from p7 import Grammar, ConstrainedGenerator
from p7.models import list_models
from services.generation import build_vocab, generate_constrained_stream, generate_unconstrained_stream, get_active_generations, get_model_and_tokenizer
from services.grammar import check_partial_completable, validate_grammar, GrammarValidationResult
from services.models import get_device_info

app = Flask(__name__)
CORS(app)
START_TIME = time.time()

logging.basicConfig(level=os.getenv("P7_LOG_LEVEL", "INFO"))
logger = logging.getLogger("p7-backend")




@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    device_info = get_device_info()
    return jsonify({
        "status": "ok",
        "version": p7.__version__,
        **device_info,
    })


@app.route('/api/debug/server', methods=['GET'])
def debug_server():
    return jsonify({
        "pid": os.getpid(),
        "ppid": os.getppid(),
        "threads": threading.active_count(),
        "uptime_seconds": int(time.time() - START_TIME),
        "server_software": os.environ.get("SERVER_SOFTWARE"),
        "gunicorn_cmd_args": os.environ.get("GUNICORN_CMD_ARGS"),
    })


@app.route('/api/debug/generations', methods=['GET'])
def debug_generations():
    return jsonify({
        "active_generations": get_active_generations(),
        "count": len(get_active_generations()),
    })


@app.route('/api/models', methods=['GET'])
def model_list():
    return jsonify({"models": list_models()})


@app.route('/api/grammars', methods=['GET'])
def list_grammars():
    """List all available built-in grammars."""
    grammars = []
    for name in p7.list_grammars():
        info = p7.get_grammar_info(name)
        grammars.append({
            "name": name,
            "display_name": info.get("name", name),
            "description": info.get("description", ""),
            "short": info.get("short", name),
        })
    return jsonify({"grammars": grammars})


@app.route('/api/grammars/<name>', methods=['GET'])
def get_grammar(name: str):
    """Get a built-in grammar spec by name."""
    try:
        spec = p7.get_grammar(name)
        info = p7.get_grammar_info(name)
        return jsonify({
            "name": name,
            "spec": spec,
            "info": info
        })
    except ValueError as e:
        return jsonify({"error": str(e)}), 404


@app.route('/api/validate-grammar', methods=['POST'])
def validate_grammar_endpoint():
    """Validate a grammar spec."""
    data = request.get_json()
    if not data or 'spec' not in data:
        return jsonify({"error": "Missing 'spec' field"}), 400
    
    spec = data['spec']
    result = validate_grammar(spec)
    return jsonify(asdict(result))


@app.route('/api/check-partial', methods=['POST'])
def check_partial_endpoint():
    data = request.get_json()
    if not data or 'spec' not in data:
        return jsonify({"error": "Missing 'spec' field"}), 400

    spec = data['spec']
    text = data.get('input', '')
    ok, reason = check_partial_completable(spec, text)
    return jsonify({"valid": ok, "reason": reason})


@app.route('/api/debug/grammar', methods=['POST'])
def debug_grammar():
    """Debug a grammar at a given input state."""
    data = request.get_json()
    if not data or 'spec' not in data:
        return jsonify({"error": "Missing 'spec' field"}), 400
    
    spec = data['spec']
    input_text = data.get('input', '')
    
    # Validate grammar first
    validation = validate_grammar(spec)
    if not validation.valid:
        return jsonify({
            "valid": False,
            "errors": validation.errors
        })
    
    try:
        grammar = Grammar(spec)
        generator = ConstrainedGenerator(grammar)
        
        # Feed input if provided
        if input_text:
            try:
                generator.feed_raw(input_text)
            except TypeError as e:
                return jsonify({
                    "valid": True,
                    "type_error": str(e),
                    "current_text": input_text,
                    "is_complete": False,
                    "completions": {"patterns": [], "examples": []},
                    "well_typed_tree_count": 0
                })
        
        # Get debug info
        debug_info = generator.debug_completions()
        well_typed_count = generator.well_typed_tree_count()
        
        return jsonify({
            "valid": True,
            "current_text": generator.current_text(),
            "is_complete": generator.is_complete(),
            "completions": debug_info,
            "well_typed_tree_count": well_typed_count,
            "type_error": None
        })
    except BaseException as e:
        logger.exception("debug_grammar failed")
        return jsonify({
            "valid": False,
            "errors": [str(e), traceback.format_exc()]
        }), 500


@app.route('/api/debug/completions', methods=['POST'])
def debug_completions():
    """Return raw completions and debug state for a prefix."""
    data = request.get_json()
    if not data or 'spec' not in data:
        return jsonify({"error": "Missing 'spec' field"}), 400

    spec = data['spec']
    input_text = data.get('input', '')

    try:
        grammar = Grammar(spec)
        generator = ConstrainedGenerator(grammar)

        if input_text:
            try:
                generator.feed_raw(input_text)
            except TypeError as e:
                return jsonify({
                    "valid": True,
                    "type_error": str(e),
                    "current_text": input_text,
                    "is_complete": False,
                    "completions": [],
                    "debug_completions": {"patterns": [], "examples": []},
                    "well_typed_tree_count": 0
                })

        return jsonify({
            "valid": True,
            "current_text": generator.current_text(),
            "is_complete": generator.is_complete(),
            "completions": generator.get_completions(),
            "debug_completions": generator.debug_completions(),
            "well_typed_tree_count": generator.well_typed_tree_count(),
            "type_error": None
        })
    except BaseException as e:
        logger.exception("debug_completions failed")
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500


@app.route('/api/debug/token-filter', methods=['POST'])
def debug_token_filter():
    """Check how grammar completions map to model vocab tokens."""
    data = request.get_json()
    if not data or 'spec' not in data or 'model' not in data:
        return jsonify({"error": "Missing 'spec' or 'model' field"}), 400

    spec = data['spec']
    input_text = data.get('input', '')
    model_name = data['model']

    try:
        tokenizer, _ = get_model_and_tokenizer(model_name)
        vocab = build_vocab(tokenizer)
        grammar = Grammar(spec)
        generator = ConstrainedGenerator(grammar)

        if input_text:
            generator.feed_raw(input_text)

        completions = generator.get_completions()
        valid_indices = generator.filter_completion_indices(vocab)
        sample = [vocab[i] for i in list(valid_indices)[:30]]
        vocab_set = set(vocab)
        completion_checks = []
        for completion in completions[:20]:
            try:
                token_ids = tokenizer.encode(completion, add_special_tokens=False)
            except TypeError:
                token_ids = tokenizer.encode(completion)
            decoded = []
            for token_id in token_ids:
                try:
                    decoded.append(tokenizer.decode([token_id]))
                except Exception:
                    decoded.append(None)
            completion_checks.append({
                "completion": completion,
                "in_vocab": completion in vocab_set,
                "token_ids": token_ids,
                "decoded": decoded,
            })

        return jsonify({
            "valid": True,
            "current_text": generator.current_text(),
            "is_complete": generator.is_complete(),
            "completion_count": len(completions),
            "valid_token_count": len(valid_indices),
            "valid_token_sample": sample,
            "completion_checks": completion_checks,
            "vocab_size": len(vocab),
        })
    except TypeError as e:
        return jsonify({"valid": True, "type_error": str(e)}), 200
    except BaseException as e:
        logger.exception("debug_token_filter failed")
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500


@app.route('/api/get-completions', methods=['POST'])
def get_completions():
    """Get valid completions for a given input state."""
    data = request.get_json()
    if not data or 'spec' not in data:
        return jsonify({"error": "Missing 'spec' field"}), 400
    
    spec = data['spec']
    input_text = data.get('input', '')
    
    try:
        grammar = Grammar(spec)
        generator = ConstrainedGenerator(grammar)
        
        if input_text:
            generator.feed_raw(input_text)
        
        completions = generator.get_completions()
        
        return jsonify({
            "current_text": generator.current_text(),
            "completions": completions,
            "is_complete": generator.is_complete()
        })
    except BaseException as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/generate-constrained', methods=['POST'])
def generate_constrained():
    """Stream constrained generation. Prompt is treated as the model context."""
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    spec = data.get('spec')
    prompt = data.get('prompt', '')
    initial = data.get('initial', '')
    model_name = data.get('model', 'gpt2')
    max_tokens = data.get('max_tokens', 50)
    grammar_tokens_raw = data.get('grammar_tokens')
    try:
        grammar_tokens = int(grammar_tokens_raw) if grammar_tokens_raw is not None else None
    except (TypeError, ValueError):
        grammar_tokens = None
    stop_on_complete = data.get('stop_on_complete', False)
    mask_whitespace = data.get('mask_whitespace', True)

    if not spec:
        return jsonify({"error": "Missing 'spec' field"}), 400

    return Response(
        generate_constrained_stream(
            spec=spec,
            prompt=prompt,
            initial=initial,
            model_name=model_name,
            max_tokens=max_tokens,
            grammar_tokens=grammar_tokens,
            stop_on_complete=stop_on_complete,
            mask_whitespace=mask_whitespace,
        ),
        mimetype='text/event-stream'
    )


@app.route('/api/generate-unconstrained', methods=['POST'])
def generate_unconstrained():
    """Stream unconstrained generation for a raw prompt."""
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    prompt = data.get('prompt', '')
    model_name = data.get('model', 'gpt2')
    max_tokens = data.get('max_tokens', 50)
    top_k = data.get('top_k', 50)
    temperature = data.get('temperature', 1.0)

    if not prompt:
        return jsonify({"error": "Missing 'prompt' field"}), 400

    return Response(
        generate_unconstrained_stream(
            prompt=prompt,
            model_name=model_name,
            max_tokens=max_tokens,
            top_k=top_k,
            temperature=temperature,
        ),
        mimetype='text/event-stream'
    )




@app.route('/api/parse-to-ast', methods=['POST'])
def parse_to_ast():
    """Parse input and return AST representation."""
    data = request.get_json()
    if not data or 'spec' not in data:
        return jsonify({"error": "Missing 'spec' field"}), 400
    
    spec = data['spec']
    input_text = data.get('input', '')
    
    try:
        grammar = Grammar(spec)
        generator = ConstrainedGenerator(grammar)
        
        if input_text:
            generator.feed_raw(input_text)
        
        try:
            sexpr = generator.to_sexpr()
            return jsonify({
                "success": True,
                "sexpr": sexpr,
                "current_text": generator.current_text(),
                "is_complete": generator.is_complete()
            })
        except Exception as e:
            return jsonify({
                "success": False,
                "error": str(e),
                "current_text": generator.current_text(),
                "is_complete": generator.is_complete()
            })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
