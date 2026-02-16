import React, { useState, useRef, useCallback, useEffect } from 'react';
import styled from 'styled-components';
import { API_BASE_URL } from '../config';

const PanelContainer = styled.div`
  height: 300px;
  display: flex;
  flex-direction: column;
  border-top: 1px solid #e5e7eb;
  background-color: #ffffff;
`;

const PanelHeader = styled.div`
  padding: 10px 15px;
  background-color: #ffffff;
  border-bottom: 1px solid #e5e7eb;
  display: flex;
  justify-content: space-between;
  align-items: center;
`;

const PanelTitle = styled.span`
  font-weight: 600;
  font-size: 14px;
  color: #111827;
`;

const ConfigRow = styled.div`
  display: flex;
  gap: 15px;
  align-items: center;
`;

const ConfigItem = styled.div`
  display: flex;
  align-items: center;
  gap: 5px;
`;

const ConfigLabel = styled.label`
  font-size: 11px;
  color: #6b7280;
  text-transform: uppercase;
`;

const Input = styled.input`
  padding: 5px 8px;
  background-color: #ffffff;
  border: 1px solid #e5e7eb;
  color: #111827;
  border-radius: 3px;
  font-size: 12px;
  
  &:focus {
    outline: none;
    border-color: #0e639c;
  }
`;

const Select = styled.select`
  padding: 5px 8px;
  background-color: #ffffff;
  border: 1px solid #e5e7eb;
  color: #111827;
  border-radius: 3px;
  font-size: 12px;
`;

const Button = styled.button`
  padding: 6px 15px;
  background-color: ${props => props.variant === 'secondary' ? '#f3f4f6' : '#0e639c'};
  border: 1px solid ${props => props.variant === 'secondary' ? '#e5e7eb' : '#0e639c'};
  color: ${props => props.variant === 'secondary' ? '#111827' : 'white'};
  border-radius: 3px;
  cursor: pointer;
  font-size: 12px;
  font-weight: 600;
  
  &:hover {
    background-color: ${props => props.variant === 'secondary' ? '#e5e7eb' : '#1177bb'};
  }
  
  &:disabled {
    background-color: #e5e7eb;
    border-color: #e5e7eb;
    cursor: not-allowed;
  }
`;

const ContentArea = styled.div`
  flex: 1;
  display: flex;
  overflow: hidden;
`;

const PromptSection = styled.div`
  width: 35%;
  padding: 15px;
  border-right: 1px solid #e5e7eb;
  display: flex;
  flex-direction: column;
  gap: 10px;
`;

const SectionTitle = styled.label`
  font-size: 11px;
  color: #6b7280;
  text-transform: uppercase;
  font-weight: 600;
`;

const TextArea = styled.textarea`
  flex: 1;
  padding: 10px;
  background-color: #ffffff;
  border: 1px solid #e5e7eb;
  color: #111827;
  border-radius: 3px;
  font-family: 'Consolas', monospace;
  font-size: 13px;
  resize: none;
  
  &:focus {
    outline: none;
    border-color: #0e639c;
  }
  
  &::placeholder {
    color: #9ca3af;
  }
`;

const OutputSection = styled.div`
  width: 65%;
  display: flex;
`;

const OutputPane = styled.div`
  flex: 1;
  padding: 15px;
  border-right: ${props => props.border ? '1px solid #e5e7eb' : 'none'};
  display: flex;
  flex-direction: column;
  gap: 10px;
  background-color: ${props => props.highlight ? '#ecfdf5' : '#ffffff'};
`;

const OutputLabel = styled.div`
  font-size: 11px;
  font-weight: 600;
  text-transform: uppercase;
  color: ${props => props.color || '#6b7280'};
  display: flex;
  justify-content: space-between;
  align-items: center;
`;

const OutputBox = styled.div`
  flex: 1;
  padding: 12px;
  background-color: #ffffff;
  border: 1px solid #e5e7eb;
  border-radius: 3px;
  font-family: 'Consolas', monospace;
  font-size: 13px;
  overflow: auto;
  white-space: pre-wrap;
  word-break: break-all;
  color: #111827;
  line-height: 1.5;
`;

const StatusBadge = styled.span`
  padding: 2px 8px;
  border-radius: 3px;
  font-size: 10px;
  font-weight: 600;
  background-color: ${props => {
    switch(props.status) {
      case 'complete': return '#ecfdf5';
      case 'generating': return '#eef2ff';
      case 'error': return '#fff1f2';
      default: return '#f3f4f6';
    }
  }};
  color: ${props => {
    switch(props.status) {
      case 'complete': return '#065f46';
      case 'generating': return '#1e3a8a';
      case 'error': return '#7f1d1d';
      default: return '#6b7280';
    }
  }};
`;

const LoadingStatus = styled.div`
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 11px;
  color: #1e3a8a;
  min-height: 16px;
`;

const Spinner = styled.span`
  width: 12px;
  height: 12px;
  border: 2px solid #e5e7eb;
  border-top-color: #0ea5a0;
  border-radius: 50%;
  display: inline-block;
  animation: spin 0.9s linear infinite;

  @keyframes spin {
    to {
      transform: rotate(360deg);
    }
  }
`;

const TokenStream = styled.div`
  font-size: 11px;
  color: #0f766e;
  margin-top: 5px;
  font-family: 'Consolas', monospace;
`;

function GenerationPanel({ grammar, grammarValid, onDebugUpdate }) {
  const [prompt, setPrompt] = useState('Complete this expression:\n');
  const [initial, setInitial] = useState('');
  const [model, setModel] = useState('gpt2');
  const [availableModels, setAvailableModels] = useState([]);
  const [grammarTokens, setGrammarTokens] = useState(30);
  const [stopOnComplete, setStopOnComplete] = useState(true);
  const [maskWhitespace, setMaskWhitespace] = useState(true);
  const [unconstrainedTopK, setUnconstrainedTopK] = useState(50);
  const [unconstrainedTemperature, setUnconstrainedTemperature] = useState(1.0);
  const [isGenerating, setIsGenerating] = useState(false);
  
  const [constrainedOutput, setConstrainedOutput] = useState('');
  const [unconstrainedOutput, setUnconstrainedOutput] = useState('');
  const [constrainedTokens, setConstrainedTokens] = useState([]);
  const [unconstrainedTokens, setUnconstrainedTokens] = useState([]);
  const [status, setStatus] = useState('idle');
  const [stoppedReason, setStoppedReason] = useState('');
  const [isComplete, setIsComplete] = useState(false);
  const [loadingMessage, setLoadingMessage] = useState('');
  const [unconstrainedStatus, setUnconstrainedStatus] = useState(null);

  const abortControllerRef = useRef(null);

  useEffect(() => {
    fetch(`${API_BASE_URL}/api/models`)
      .then(res => res.json())
      .then(data => {
        const models = data.models || [];
        setAvailableModels(models);
        if (models.length && !models.find((m) => m.name === model)) {
          setModel(models[0].name);
        }
      })
      .catch(err => console.error('Failed to load models:', err));
  }, [model]);

  const generate = useCallback(async () => {
    if (!grammarValid || isGenerating) return;

    setIsGenerating(true);
    setConstrainedOutput(initial);
    setUnconstrainedOutput('');
    setConstrainedTokens([]);
    setUnconstrainedTokens([]);
    setStatus('generating');
    setStoppedReason('');
    setIsComplete(false);
    setLoadingMessage('Starting unconstrained generation...');
    setUnconstrainedStatus(null);

    abortControllerRef.current = new AbortController();

    const readStream = async (response, onEvent) => {
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const events = buffer.split('\n\n');
        buffer = events.pop() || '';

        for (const event of events) {
          const lines = event.split('\n');
          const dataLines = lines
            .filter((line) => line.startsWith('data: '))
            .map((line) => line.slice(6));

          if (dataLines.length === 0) continue;

          try {
            const data = JSON.parse(dataLines.join('\n'));
            onEvent(data);
          } catch (e) {
            // Skip malformed events
          }
        }
      }
    };

    try {
      let unconstrainedText = '';
      let unconstrainedTokens = [];

      const unconstrainedResponse = await fetch(`${API_BASE_URL}/api/generate-unconstrained`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        signal: abortControllerRef.current.signal,
        body: JSON.stringify({
          prompt,
          model,
          max_tokens: parseInt(grammarTokens),
          top_k: parseInt(unconstrainedTopK),
          temperature: parseFloat(unconstrainedTemperature)
        })
      });

      if (!unconstrainedResponse.ok) {
        const errorData = await unconstrainedResponse.json();
        throw new Error(errorData.error || `HTTP ${unconstrainedResponse.status}`);
      }

      await readStream(unconstrainedResponse, (data) => {
        switch (data.type) {
          case 'status':
            setLoadingMessage(data.message);
            break;
          case 'token':
            unconstrainedText = data.full_text || unconstrainedText + (data.text || '');
            if (data.text) {
              unconstrainedTokens.push(data.text);
            }
            setUnconstrainedOutput(unconstrainedText);
            setUnconstrainedTokens([...unconstrainedTokens]);
            break;
          case 'done':
            setLoadingMessage('Starting constrained generation...');
            if (unconstrainedText.trim()) {
              checkUnconstrained(unconstrainedText);
            }
            break;
          case 'error':
            throw new Error(data.message);
          default:
            break;
        }
      });

      let constrainedText = initial;
      let constrainedTokens = [];
      let stoppedReason = 'max_tokens';
      let isComplete = false;

      const constrainedResponse = await fetch(`${API_BASE_URL}/api/generate-constrained`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        signal: abortControllerRef.current.signal,
        body: JSON.stringify({
          spec: grammar,
          prompt,
          initial,
          model,
          max_tokens: parseInt(grammarTokens),
          grammar_tokens: parseInt(grammarTokens),
          stop_on_complete: stopOnComplete,
          mask_whitespace: maskWhitespace
        })
      });

      if (!constrainedResponse.ok) {
        const errorData = await constrainedResponse.json();
        throw new Error(errorData.error || `HTTP ${constrainedResponse.status}`);
      }

      await readStream(constrainedResponse, (data) => {
        switch (data.type) {
          case 'status':
            setLoadingMessage(data.message);
            break;
          case 'token':
            if (loadingMessage) {
              setLoadingMessage('');
            }
            constrainedText = data.full_text || constrainedText + (data.text || '');
            if (data.text) {
              constrainedTokens.push(data.text);
            }
            setConstrainedOutput(constrainedText);
            setConstrainedTokens([...constrainedTokens]);
            break;
          case 'done':
            setLoadingMessage('');
            stoppedReason = data.reason || 'max_tokens';
            isComplete = data.is_complete || false;
            setStoppedReason(stoppedReason);
            setIsComplete(isComplete);
            setStatus(isComplete ? 'complete' : 'stopped');

            onDebugUpdate({
              current_text: constrainedText,
              is_complete: isComplete,
              completions: { patterns: [], examples: [] },
              well_typed_tree_count: isComplete ? 1 : 0,
              type_error: null
            });
            break;
          case 'error':
            throw new Error(data.message);
          default:
            break;
        }
      });
    } catch (err) {
      if (err.name === 'AbortError') {
        setStatus('stopped');
        setStoppedReason('cancelled');
        setLoadingMessage('');
      } else {
        setStatus('error');
        setConstrainedOutput(`Error: ${err.message}`);
        setUnconstrainedOutput(`Error: ${err.message}`);
        setLoadingMessage('');
      }
    } finally {
      setIsGenerating(false);
      abortControllerRef.current = null;
    }
  }, [grammar, grammarValid, prompt, initial, model, grammarTokens, isGenerating, onDebugUpdate, stopOnComplete, maskWhitespace, unconstrainedTopK, unconstrainedTemperature]);

  const clear = useCallback(() => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
    setConstrainedOutput('');
    setUnconstrainedOutput('');
    setConstrainedTokens([]);
    setUnconstrainedTokens([]);
    setStatus('idle');
    setStoppedReason('');
    setIsComplete(false);
    setLoadingMessage('');
    setUnconstrainedStatus(null);
  }, []);

  const checkUnconstrained = useCallback(async (text) => {
    if (!grammarValid) return;
    try {
      const response = await fetch(`${API_BASE_URL}/api/debug/grammar`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ spec: grammar, input: text })
      });
      const data = await response.json();
      setUnconstrainedStatus({
        valid: data.valid,
        is_complete: data.is_complete,
        well_typed_tree_count: data.well_typed_tree_count || 0,
        type_error: data.type_error || null
      });
    } catch (err) {
      setUnconstrainedStatus({
        valid: false,
        is_complete: false,
        well_typed_tree_count: 0,
        type_error: err.message
      });
    }
  }, [grammar, grammarValid]);

  const stopGeneration = useCallback(() => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      setStatus('stopped');
      setStoppedReason('cancelled');
      setLoadingMessage('');
    }
  }, []);

  const formatTokenStream = (tokens) => {
    if (tokens.length === 0) return '';
    return tokens.slice(-10).map(t => t.replace(/\s/g, '·')).join(' ');
  };

  return (
    <PanelContainer>
      <PanelHeader>
        <PanelTitle>Constrained Generation</PanelTitle>
        {loadingMessage && (
          <LoadingStatus>
            <Spinner />
            <span>{loadingMessage}</span>
          </LoadingStatus>
        )}
        <ConfigRow>
          <ConfigItem>
            <ConfigLabel>Model</ConfigLabel>
            <Select value={model} onChange={(e) => setModel(e.target.value)}>
              {availableModels.map((m) => (
                <option key={m.name} value={m.name}>{m.display_name}</option>
              ))}
            </Select>
          </ConfigItem>
          
          <ConfigItem>
            <ConfigLabel>Max Tokens</ConfigLabel>
            <Input
              type="number"
              value={grammarTokens}
              onChange={(e) => setGrammarTokens(e.target.value)}
              style={{ width: '70px' }}
            />
          </ConfigItem>

          <ConfigItem>
            <ConfigLabel>Top-K</ConfigLabel>
            <Input
              type="number"
              value={unconstrainedTopK}
              onChange={(e) => setUnconstrainedTopK(e.target.value)}
              style={{ width: '60px' }}
            />
          </ConfigItem>

          <ConfigItem>
            <ConfigLabel>Temp</ConfigLabel>
            <Input
              type="number"
              step="0.1"
              value={unconstrainedTemperature}
              onChange={(e) => setUnconstrainedTemperature(e.target.value)}
              style={{ width: '60px' }}
            />
          </ConfigItem>
          <ConfigItem>
            <ConfigLabel>Stop on Complete</ConfigLabel>
            <Input
              type="checkbox"
              checked={stopOnComplete}
              onChange={(e) => setStopOnComplete(e.target.checked)}
            />
          </ConfigItem>

          <ConfigItem>
            <ConfigLabel>Mask Whitespace</ConfigLabel>
            <Input
              type="checkbox"
              checked={maskWhitespace}
              onChange={(e) => setMaskWhitespace(e.target.checked)}
            />
          </ConfigItem>
          
          <Button 
            onClick={generate} 
            disabled={!grammarValid || isGenerating}
          >
            {isGenerating ? 'Generating...' : 'Generate'}
          </Button>

          <Button
            variant="secondary"
            onClick={stopGeneration}
            disabled={!isGenerating}
          >
            Stop
          </Button>
          
          <Button 
            variant="secondary" 
            onClick={clear}
            disabled={isGenerating}
          >
            Clear
          </Button>
        </ConfigRow>
      </PanelHeader>
      
      <ContentArea>
        <PromptSection>
          <SectionTitle>Prompt</SectionTitle>
          <TextArea
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            placeholder="Enter prompt for the model..."
          />
          
          <SectionTitle>Initial Seed</SectionTitle>
          <Input
            type="text"
            value={initial}
            onChange={(e) => setInitial(e.target.value)}
            placeholder="Starting text for generation..."
            style={{ width: '100%' }}
          />
        </PromptSection>
        
        <OutputSection>
          <OutputPane border highlight={isComplete}>
            <OutputLabel color="#4ec9b0">
              Constrained (Type-Safe)
              <StatusBadge status={status === 'generating' ? 'generating' : isComplete ? 'complete' : status}>
                {isComplete ? 'Complete' : status === 'generating' ? 'Generating...' : stoppedReason || 'Ready'}
              </StatusBadge>
            </OutputLabel>
            <OutputBox>{constrainedOutput}</OutputBox>
            {constrainedTokens.length > 0 && (
              <TokenStream>
                Last tokens: {formatTokenStream(constrainedTokens)}
              </TokenStream>
            )}
          </OutputPane>
          
          <OutputPane>
            <OutputLabel color="#dcdcaa">
              Unconstrained (Raw Model)
              {unconstrainedStatus ? (
                <StatusBadge status={unconstrainedStatus.valid ? (unconstrainedStatus.is_complete ? 'complete' : 'stopped') : 'error'}>
                  {unconstrainedStatus.valid
                    ? (unconstrainedStatus.is_complete ? 'Parse OK' : 'Incomplete')
                    : 'Invalid'}
                </StatusBadge>
              ) : (
                <StatusBadge status={status === 'generating' ? 'generating' : status === 'idle' ? 'idle' : 'complete'}>
                  {status === 'idle' ? 'Ready' : status === 'generating' ? 'Generating...' : 'Complete'}
                </StatusBadge>
              )}
            </OutputLabel>
            <OutputBox>{unconstrainedOutput}</OutputBox>
            {unconstrainedStatus && (
              <TokenStream>
                {unconstrainedStatus.valid
                  ? `Well-typed trees: ${unconstrainedStatus.well_typed_tree_count}`
                  : `Type error: ${unconstrainedStatus.type_error || 'parse failed'}`}
              </TokenStream>
            )}
            {unconstrainedTokens.length > 0 && (
              <TokenStream>
                Last tokens: {formatTokenStream(unconstrainedTokens)}
              </TokenStream>
            )}
          </OutputPane>
        </OutputSection>
      </ContentArea>
    </PanelContainer>
  );
}

export default GenerationPanel;
