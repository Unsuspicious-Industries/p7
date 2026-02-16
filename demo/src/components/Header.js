import React from 'react';
import styled from 'styled-components';

const HeaderContainer = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 10px 20px;
  background-color: #ffffff;
  border-bottom: 1px solid #e5e7eb;
`;

const Title = styled.h1`
  margin: 0;
  font-size: 18px;
  color: #1f2937;
`;

const Subtitle = styled.span`
  font-size: 12px;
  color: #6b7280;
  margin-left: 10px;
`;

const Controls = styled.div`
  display: flex;
  gap: 10px;
  align-items: center;
`;

const Select = styled.select`
  padding: 5px 10px;
  background-color: #ffffff;
  border: 1px solid #e5e7eb;
  color: #1f2937;
  border-radius: 3px;
  font-size: 12px;
`;

const StatusBadge = styled.span`
  padding: 4px 10px;
  border-radius: 3px;
  font-size: 12px;
  font-weight: 600;
  background-color: ${props => props.valid ? '#ecfdf5' : '#fff1f2'};
  color: ${props => props.valid ? '#065f46' : '#991b1b'};
`;

const DeviceBadge = styled.span`
  padding: 4px 8px;
  border-radius: 999px;
  font-size: 11px;
  background-color: rgba(59, 130, 246, 0.12);
  color: #1d4ed8;
  border: 1px solid rgba(59, 130, 246, 0.25);
`;

function Header({ grammarStatus, availableGrammars, onLoadExample, isLoading, deviceInfo }) {
  const deviceLabel = deviceInfo?.device === 'cuda'
    ? `GPU: ${deviceInfo?.gpu_name || 'CUDA'}`
    : 'CPU';

  return (
    <HeaderContainer>
      <div>
        <Title>P7 Visualization Platform</Title>
        <Subtitle>Grammar Editor & Constrained Generation Workflows</Subtitle>
      </div>
      
      <Controls>
        <DeviceBadge>{deviceLabel}</DeviceBadge>
        <span style={{ fontSize: '12px', color: '#808080' }}>Load Example:</span>
        <Select 
          onChange={(e) => e.target.value && onLoadExample(e.target.value)}
          value=""
          disabled={isLoading}
        >
          <option value="">{isLoading ? 'Loading...' : 'Select...'}</option>
          {availableGrammars.map(g => (
            <option key={g.name} value={g.name}>
              {g.display_name}
            </option>
          ))}
        </Select>
        
        <StatusBadge valid={grammarStatus.valid}>
          {grammarStatus.valid ? 'Grammar Valid' : 'Grammar Invalid'}
        </StatusBadge>
      </Controls>
    </HeaderContainer>
  );
}

export default Header;
