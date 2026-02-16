import React from 'react';
import Editor from 'react-simple-code-editor';
import styled from 'styled-components';

const EditorContainer = styled.div`
  flex: 1;
  overflow: auto;
  background-color: #ffffff;
  border: 1px solid #e5e7eb;

  & textarea {
    font-family: 'Consolas', 'Monaco', 'Courier New', monospace !important;
    font-size: 14px !important;
    line-height: 1.5 !important;
    color: #1f2937 !important;
  }
`;

const ErrorList = styled.div`
  max-height: 150px;
  overflow: auto;
  background-color: #fff5f5;
  border-top: 1px solid #fecaca;
  padding: 10px;
`;

const ErrorItem = styled.div`
  color: #b91c1c;
  font-size: 12px;
  margin-bottom: 5px;
  padding: 5px;
  background-color: #fee2e2;
  border-radius: 3px;
`;

// Simple syntax highlighting for .spec files
const highlight = (code) => {
  // Return code as-is - newlines will be preserved by the editor
  // In the future, could add syntax highlighting here by wrapping tokens in spans
  return code;
};

function GrammarEditor({ value, onChange, errors}) {
  return (
    <>
      <EditorContainer>
        <Editor
          value={value}
          onValueChange={onChange}
          highlight={highlight}
          padding={20}
          style={{
            fontFamily: '"Consolas", "Monaco", "Courier New", monospace',
            fontSize: 14,
            backgroundColor: '#ffffff',
            color: '#1f2937',
            minHeight: '100%',
          }}
          textareaClassName="grammar-editor"
        />
      </EditorContainer>
      
      {errors && errors.length > 0 && (
        <ErrorList>
          {errors.map((error, idx) => (
            <ErrorItem key={idx}>{error}</ErrorItem>
          ))}
        </ErrorList>
      )}
    </>
  );
}

export default GrammarEditor;
