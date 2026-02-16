import React from 'react';
import styled, { createGlobalStyle } from 'styled-components';

const GlobalStyle = createGlobalStyle`
  @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=JetBrains+Mono:wght@400;600&display=swap');
  body {
    margin: 0;
    font-family: 'Space Grotesk', sans-serif;
    background: #f8fafc;
    color: #1f2937;
  }
`;

const Page = styled.div`
  min-height: 100vh;
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
  padding: 40px;
  text-align: center;
  background: #f8fafc;
`;

const Title = styled.h1`
  font-size: 36px;
  font-weight: 700;
  margin-bottom: 12px;
  letter-spacing: 0.2px;
  color: #1f2937;
`;

const Subtitle = styled.p`
  max-width: 720px;
  font-size: 16px;
  color: #6b7280;
  margin-bottom: 28px;
`;

const CardRow = styled.div`
  display: grid;
  grid-template-columns: repeat(2, minmax(220px, 1fr));
  gap: 16px;
  width: min(820px, 100%);
`;

const Card = styled.a`
  padding: 18px 16px;
  border-radius: 12px;
  border: 1px solid #e2e8f0;
  background: #ffffff;
  color: #1f2937;
  text-decoration: none;
  display: flex;
  flex-direction: column;
  gap: 8px;
  box-shadow: 0 8px 20px rgba(31, 41, 55, 0.06);
  transition: transform 0.15s ease, border 0.15s ease, box-shadow 0.15s ease;

  &:hover {
    transform: translateY(-2px);
    border-color: #3730a3;
    box-shadow: 0 12px 24px rgba(55, 48, 163, 0.12);
  }
`;

const CardTitle = styled.div`
  font-weight: 600;
  color: #1f2937;
`;

const CardDesc = styled.div`
  font-size: 13px;
  color: #6b7280;
`;

const QuoteCard = styled.div`
  margin: 18px 0 24px 0;
  padding: 16px 18px;
  border-radius: 12px;
  border: 1px solid #e5e7eb;
  background: #ffffff;
  box-shadow: 0 10px 20px rgba(31, 41, 55, 0.06);
  font-family: 'Fraunces', serif;
  font-size: 16px;
  color: #1f2937;
  line-height: 1.5;
  max-width: 680px;
`;

const QuoteAttribution = styled.div`
  margin-top: 8px;
  font-size: 12px;
  color: #6b7280;
  font-family: 'Space Grotesk', sans-serif;
`;

const Section = styled.div`
  max-width: 860px;
  margin: 0 auto 26px auto;
  text-align: left;
  display: grid;
  gap: 14px;
`;

const SectionTitle = styled.h2`
  font-size: 18px;
  margin: 0;
  color: #1f2937;
`;

const SectionText = styled.p`
  margin: 0;
  color: #6b7280;
  line-height: 1.6;
`;

const Columns = styled.div`
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 16px;
`;

const BulletList = styled.ul`
  margin: 0;
  padding-left: 18px;
  color: #6b7280;
  line-height: 1.6;
`;

const CodeBox = styled.div`
  border: 1px solid #e5e7eb;
  background: #ffffff;
  padding: 12px 14px;
  border-radius: 10px;
  font-family: 'JetBrains Mono', monospace;
  font-size: 16px;
  color: #1f2937;
  box-shadow: 0 6px 16px rgba(31, 41, 55, 0.06);
  white-space: pre-wrap;
`;

function HomePage() {
  return (
    <>
      <GlobalStyle />
      <Page>
        <Title>Proposition 7: Type‑Safe Constrained Generation</Title>
        <Subtitle>
          P7 lets you generate only well‑typed programs and structured text. Explore the demo,
          dive into the lab, or read the background and spec. 
        </Subtitle>
        <QuoteCard>
          “Whereof one cannot speak, thereof one must be silent”
          <QuoteAttribution>— Ludwig Wittgenstein</QuoteAttribution>
        </QuoteCard>
        <Section>
          <SectionTitle>What P7 Does</SectionTitle>
          <SectionText>
            P7 is a constrained generation engine that only emits programs that are syntactically valid and
            well‑typed. It builds partial trees for every prefix, checks them against declarative typing rules,
            and restricts next‑token choices to completions that preserve type safety.
          </SectionText>
        </Section>
        <Section>
          <SectionTitle>How It Works (From the Spec)</SectionTitle>
          <Columns>
            <div>
              <SectionText>
                The grammar defines productions and bindings. Typing rules declare constraints on the
                bound subtrees, and P7 computes valid next tokens from the FIRST sets of the frontier.
              </SectionText>
              <BulletList>
                <li>Partial parser produces a forest of incomplete ASTs.</li>
                <li>Typing rules filter trees to only well‑typed branches.</li>
                <li>Completion tokens are the only allowed next tokens.</li>
              </BulletList>
            </div>
            <CodeBox>{`Completability: s is completable iff ∃ s'. ss' ∈ L

Typing rule:
Γ ⊢ e : τ
----------- (rule)
   τ

Next tokens = FIRST(frontier)`
}</CodeBox>
          </Columns>
        </Section>
        <CardRow>
          <Card href="/demo">
            <CardTitle>Demo</CardTitle>
            <CardDesc>Simple dual‑mode chat UI: thinking + constrained output.</CardDesc>
          </Card>
          <Card href="/lab">
            <CardTitle>Lab</CardTitle>
            <CardDesc>Full grammar editor, debugger, and generator tools.</CardDesc>
          </Card>
          <Card href="https://unsuspicious.org/blog/proposition-7" target="_blank" rel="noreferrer">
            <CardTitle>Blog Post</CardTitle>
            <CardDesc>Why P7 exists and the core idea behind typed constraints. The Curry-Howard twin of this platfom.</CardDesc>
          </Card>
          <Card href="https://unsuspicious-industries.github.io/p7/" target="_blank" rel="noreferrer">
            <CardTitle>Spec</CardTitle>
            <CardDesc>Formal definition, grammar format, and typing rules.</CardDesc>
          </Card>
        </CardRow>
      </Page>
    </>
  );
}

export default HomePage;
