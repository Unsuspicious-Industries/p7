import React, { useCallback, useEffect, useRef, useState } from 'react';
import styled, { createGlobalStyle } from 'styled-components';
import GrammarEditor from '../components/GrammarEditor';
import { API_BASE_URL } from '../config';

const GlobalStyle = createGlobalStyle`
  @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=Fraunces:opsz,wght@9..144,400;9..144,600&family=JetBrains+Mono:wght@400;600&display=swap');
  body {
    margin: 0;
    font-family: 'Space Grotesk', sans-serif;
    background: #f8fafc;
    overflow: hidden;
  }

  .grammar-editor {
    scrollbar-width: thin;
    scrollbar-color: #94a3b8 transparent;
  }

  .grammar-editor::-webkit-scrollbar {
    width: 10px;
  }

  .grammar-editor::-webkit-scrollbar-track {
    background: transparent;
  }

  .grammar-editor::-webkit-scrollbar-thumb {
    background: #cbd5f5;
    border-radius: 999px;
    border: 2px solid transparent;
    background-clip: padding-box;
  }
`;

const Page = styled.div`
  height: 100vh;
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 0;
  background:
    radial-gradient(900px 600px at 5% 5%, rgba(55, 48, 163, 0.08), transparent 60%),
    radial-gradient(800px 500px at 95% 10%, rgba(6, 182, 212, 0.08), transparent 55%),
    #f8fafc;
  color: #1f2937;
`;

const Panel = styled.div`
  display: flex;
  flex-direction: column;
  border-right: ${props => props.noBorder ? 'none' : '1px solid #e5e7eb'};
  min-height: 0;
`;

const PanelHeader = styled.div`
  padding: 18px 20px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  border-bottom: 1px solid #e5e7eb;
  background: #ffffff;
`;


const Title = styled.div`
  font-weight: 600;
  letter-spacing: 0.2px;
  color: #1f2937;
`;

const Badge = styled.span`
  font-size: 11px;
  padding: 4px 8px;
  border-radius: 999px;
  background: ${props => props.ok ? 'rgba(16, 185, 129, 0.15)' : 'rgba(239, 68, 68, 0.12)'};
  color: ${props => props.ok ? '#10b981' : '#ef4444'};
  border: 1px solid ${props => props.ok ? 'rgba(16, 185, 129, 0.3)' : 'rgba(239, 68, 68, 0.25)'};
`;

const Select = styled.select`
  padding: 6px 10px;
  border-radius: 6px;
  background: #ffffff;
  border: 1px solid #d1d5db;
  color: #1f2937;
  font-size: 12px;
`;

const EditorWrap = styled.div`
  flex: 1;
  overflow: hidden;
`;

const ChatWrap = styled.div`
  display: flex;
  flex-direction: column;
  height: 100%;
  min-height: 0;
`;

const Messages = styled.div`
  flex: 1;
  overflow-y: auto;
  padding: 18px;
  display: flex;
  flex-direction: column;
  gap: 16px;
  min-height: 0;
`;

const MessageCard = styled.div`
  background: #ffffff;
  border: 1px solid #e5e7eb;
  border-radius: 12px;
  padding: 14px 16px;
  box-shadow: 0 10px 20px rgba(31, 41, 55, 0.06);
`;

const MessageHeader = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 10px;
  font-size: 12px;
  color: #6b7280;
`;

const Block = styled.div`
  margin-top: 8px;
  padding: 10px 12px;
  border-radius: 10px;
  background: #f8fafc;
  border: 1px solid #e5e7eb;
  font-family: 'JetBrains Mono', monospace;
  white-space: pre-wrap;
  color: #1f2937;
  min-height: 36px;
`;

const BlockLabel = styled.div`
  font-size: 11px;
  letter-spacing: 0.3px;
  text-transform: uppercase;
  color: #6b7280;
`;

const Composer = styled.div`
  border-top: 1px solid #e5e7eb;
  padding: 12px 16px;
  display: grid;
  grid-template-columns: 1fr auto;
  gap: 10px;
  align-items: center;
  background: #ffffff;
`;

const TextArea = styled.textarea`
  background: #ffffff;
  border: 1px solid #d1d5db;
  color: #1f2937;
  border-radius: 10px;
  padding: 10px 12px;
  min-height: 60px;
  font-family: 'JetBrains Mono', monospace;
  resize: vertical;
`;

const Controls = styled.div`
  display: flex;
  flex-direction: column;
  gap: 8px;
`;

const Button = styled.button`
  padding: 8px 14px;
  border-radius: 8px;
  border: 1px solid ${props => props.secondary ? '#d1d5db' : '#3730a3'};
  background: ${props => props.secondary ? '#ffffff' : '#3730a3'};
  color: ${props => props.secondary ? '#1f2937' : '#ffffff'};
  font-size: 12px;
  cursor: pointer;
  &:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }
`;

const StatusLine = styled.div`
  font-size: 12px;
  color: #6b7280;
  padding: 8px 16px 0 16px;
`;

const DeviceBadge = styled.span`
  font-size: 11px;
  padding: 4px 8px;
  border-radius: 999px;
  background: rgba(59, 130, 246, 0.12);
  color: #1d4ed8;
  border: 1px solid rgba(59, 130, 246, 0.25);
`;

const DEFAULT_GRAMMAR = "";

function DemoPage() {
  const [grammar, setGrammar] = useState(DEFAULT_GRAMMAR);
  const [grammarStatus, setGrammarStatus] = useState({ valid: true, errors: [] });
  const [availableGrammars, setAvailableGrammars] = useState([]);
  const [selectedGrammar, setSelectedGrammar] = useState('');
  const [prompt, setPrompt] = useState('Create a playful typed phrase.');
  const [model, setModel] = useState('gpt2');
  const [thinkTokens, setThinkTokens] = useState(128);
  const [grammarTokens, setGrammarTokens] = useState(48);
  const [thinkTopK, setThinkTopK] = useState(50);
  const [thinkTemperature, setThinkTemperature] = useState(1.0);
  const [messages, setMessages] = useState([]);
  const [status, setStatus] = useState('Ready');
  const [isGenerating, setIsGenerating] = useState(false);
  const [availableModels, setAvailableModels] = useState([]);
  const [lastConstrained, setLastConstrained] = useState('');
  const [deviceInfo, setDeviceInfo] = useState({ device: 'cpu', gpu_name: '' });
  const abortRef = useRef(null);

  const validateGrammar = useCallback(async (spec) => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/validate-grammar`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ spec })
      });
      const data = await response.json();
      setGrammarStatus(data);
    } catch (err) {
      setGrammarStatus({ valid: false, errors: [err.message] });
    }
  }, []);

  const loadExample = useCallback(async (name) => {
    setSelectedGrammar(name);
    try {
      const response = await fetch(`${API_BASE_URL}/api/grammars/${name}`);
      const data = await response.json();
      if (data.auf) {
        setGrammar(data.auf);
      }
    } catch (err) {
      console.error('Failed to load grammar:', err);
    }
  }, []);

  useEffect(() => {
    fetch(`${API_BASE_URL}/api/grammars`)
      .then(res => res.json())
      .then(data => {
        const grammars = data.grammars || [];
        setAvailableGrammars(grammars);
        if (!selectedGrammar) {
          const hasToy = grammars.find((g) => g.name === 'toy');
          if (hasToy) {
            loadExample('toy');
          }
        }
      })
      .catch(err => console.error('Failed to load grammars:', err));
  }, [loadExample, selectedGrammar]);

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

  useEffect(() => {
    fetch(`${API_BASE_URL}/api/health`)
      .then(res => res.json())
      .then(data => {
        setDeviceInfo({
          device: data.device || 'cpu',
          gpu_name: data.gpu_name || '',
        });
      })
      .catch(() => setDeviceInfo({ device: 'cpu', gpu_name: '' }));
  }, []);

  useEffect(() => {
    const timeoutId = setTimeout(() => {
      validateGrammar(grammar);
    }, 400);
    return () => clearTimeout(timeoutId);
  }, [grammar]);

  const startGeneration = useCallback(async (initialOverride = '') => {
    if (!grammarStatus.valid || isGenerating) return;
    setIsGenerating(true);
    setStatus('Generating...');

    const messageId = Date.now();
    setMessages([
      {
        id: messageId,
        user: prompt,
        thinking: '...',
        constrained: '',
        state: 'streaming'
      }
    ]);

    abortRef.current = new AbortController();

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
          } catch (err) {
            // ignore malformed events
          }
        }
      }
    };

    try {
      setStatus('Generating unconstrained...');

      const unconstrainedResponse = await fetch(`${API_BASE_URL}/api/generate-unconstrained`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        signal: abortRef.current.signal,
        body: JSON.stringify({
          prompt,
          model,
          max_tokens: parseInt(thinkTokens),
          top_k: parseInt(thinkTopK),
          temperature: parseFloat(thinkTemperature)
        })
      });

      if (!unconstrainedResponse.ok) {
        const errorData = await unconstrainedResponse.json();
        throw new Error(errorData.error || `HTTP ${unconstrainedResponse.status}`);
      }

      await readStream(unconstrainedResponse, (data) => {
        if (data.type === 'status') {
          setStatus(data.message);
        }
        if (data.type === 'token') {
          const nextText = data.full_text || (data.text || '');
          setMessages((prev) => prev.map((msg) =>
            msg.id === messageId
              ? { ...msg, thinking: nextText || msg.thinking }
              : msg
          ));
        }
        if (data.type === 'done') {
          setStatus('Generating constrained...');
        }
        if (data.type === 'error') {
          throw new Error(data.message);
        }
      });

      const constrainedResponse = await fetch(`${API_BASE_URL}/api/generate-constrained`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        signal: abortRef.current.signal,
        body: JSON.stringify({
          spec: grammar,
          prompt,
          initial: initialOverride,
          model,
          max_tokens: parseInt(grammarTokens),
          grammar_tokens: parseInt(grammarTokens)
        })
      });

      if (!constrainedResponse.ok) {
        const errorData = await constrainedResponse.json();
        throw new Error(errorData.error || `HTTP ${constrainedResponse.status}`);
      }

      await readStream(constrainedResponse, (data) => {
        if (data.type === 'status') {
          setStatus(data.message);
        }
        if (data.type === 'token') {
          const nextText = data.full_text || (data.text || '');
          setMessages((prev) => prev.map((msg) =>
            msg.id === messageId
              ? { ...msg, constrained: nextText || msg.constrained }
              : msg
          ));
          if (data.full_text) {
            setLastConstrained(data.full_text);
          }
        }
        if (data.type === 'done') {
          setStatus(data.reason === 'complete' ? 'Complete' : 'Stopped');
          setMessages((prev) => prev.map((msg) =>
            msg.id === messageId ? { ...msg, state: 'done' } : msg
          ));
        }
        if (data.type === 'error') {
          throw new Error(data.message);
        }
      });
    } catch (err) {
      if (err.name === 'AbortError') {
        setStatus('Stopped');
      } else {
        setStatus('Error');
        setMessages((prev) => prev.map((msg) =>
          msg.id === messageId ? { ...msg, thinking: 'error', constrained: err.message } : msg
        ));
      }
    } finally {
      setIsGenerating(false);
      abortRef.current = null;
    }
  }, [grammar, grammarStatus.valid, isGenerating, prompt, model, thinkTokens, grammarTokens, thinkTopK, thinkTemperature]);
  

  const stopGeneration = useCallback(() => {
    if (abortRef.current) {
      abortRef.current.abort();
    }
  }, []);

  const clearMessages = useCallback(() => {
    setMessages([]);
    setStatus('Ready');
  }, []);

  return (
    <>
      <GlobalStyle />
      <Page>
        <Panel>
          <PanelHeader>
            <Title>Grammar</Title>
            <div style={{ display: 'flex', gap: '10px', alignItems: 'center' }}>
              <Select value={selectedGrammar} onChange={(e) => loadExample(e.target.value)}>
                <option value="">Load Example</option>
                {availableGrammars.map((g) => (
                  <option key={g.name} value={g.name}>{g.display_name}</option>
                ))}
              </Select>
              <Badge ok={grammarStatus.valid}>{grammarStatus.valid ? 'Valid' : 'Invalid'}</Badge>
            </div>
          </PanelHeader>
          <EditorWrap>
            <GrammarEditor value={grammar} onChange={setGrammar} errors={grammarStatus.errors} theme="light" />
          </EditorWrap>
        </Panel>
        <Panel noBorder>
          <ChatWrap>
            <PanelHeader>
              <Title>Demo Chat</Title>
              <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
                <DeviceBadge>
                  {deviceInfo.device === 'cuda'
                    ? `GPU: ${deviceInfo.gpu_name || 'CUDA'}`
                    : 'CPU'}
                </DeviceBadge>
                <Select value={model} onChange={(e) => setModel(e.target.value)}>
                  {availableModels.map((m) => (
                    <option key={m.name} value={m.name}>{m.display_name}</option>
                  ))}
                </Select>
              </div>
            </PanelHeader>
            <StatusLine>Status: {status}</StatusLine>
            <Messages>
              {messages.map((msg) => (
                <MessageCard key={msg.id}>
                  <MessageHeader>
                    <span>User</span>
                    <span>{msg.state === 'streaming' ? 'Streaming' : 'Done'}</span>
                  </MessageHeader>
                  <Block>{msg.user}</Block>
                  <div style={{ marginTop: '12px' }}>
                    <BlockLabel>Thinking</BlockLabel>
                    <Block>{msg.thinking}</Block>
                  </div>
                  <div style={{ marginTop: '12px' }}>
                    <BlockLabel>Constrained Output</BlockLabel>
                    <Block>{msg.constrained}</Block>
                  </div>
                </MessageCard>
              ))}
            </Messages>
            <Composer>
              <TextArea
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                placeholder="Ask for a typed expression..."
              />
              <Controls>
                <InputRow>
                  <Label>Thinking Tokens</Label>
                  <SmallInput
                    type="number"
                    value={thinkTokens}
                    onChange={(e) => setThinkTokens(e.target.value)}
                  />
                </InputRow>
                <InputRow>
                  <Label>Think Top-K</Label>
                  <SmallInput
                    type="number"
                    value={thinkTopK}
                    onChange={(e) => setThinkTopK(e.target.value)}
                  />
                </InputRow>
                <InputRow>
                  <Label>Think Temp</Label>
                  <SmallInput
                    type="number"
                    step="0.1"
                    value={thinkTemperature}
                    onChange={(e) => setThinkTemperature(e.target.value)}
                  />
                </InputRow>
                <InputRow>
                  <Label>Formal Tokens</Label>
                  <SmallInput
                    type="number"
                    value={grammarTokens}
                    onChange={(e) => setGrammarTokens(e.target.value)}
                  />
                </InputRow>
                <Button onClick={() => startGeneration('')} disabled={!grammarStatus.valid || isGenerating}>
                  {isGenerating ? 'Generating' : 'Generate'}
                </Button>
                <Button secondary onClick={() => startGeneration(lastConstrained)} disabled={!grammarStatus.valid || isGenerating || !lastConstrained}>
                  Continue
                </Button>
                <Button secondary onClick={stopGeneration} disabled={!isGenerating}>
                  Stop
                </Button>
                <Button secondary onClick={clearMessages} disabled={isGenerating}>
                  Clear
                </Button>
              </Controls>
            </Composer>
          </ChatWrap>
        </Panel>
      </Page>
    </>
  );
}

const InputRow = styled.div`
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 6px;
`;

const Label = styled.span`
  font-size: 11px;
  color: #6b7280;
`;

const SmallInput = styled.input`
  width: 70px;
  padding: 6px 8px;
  border-radius: 8px;
  border: 1px solid #d1d5db;
  background: #ffffff;
  color: #1f2937;
  font-size: 12px;
`;



export default DemoPage;
