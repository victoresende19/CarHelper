# DEM-1 — Novo contexto FiatArgo

> Documento gerado pela plataforma de agentes para revisão humana. Este Draft PR não realiza merge nem deploy automático.

## Demanda

Criar novo contexto FiatArgo para conseguir responder perguntas sobre o veículo. Encontrar o PDF do manual do veículo e subir na aplicação.

## Repositório-alvo

- Repositório: https://github.com/victoresende19/CarHelper
- Branch base: `main`
- Linguagem predominante: Python (100.00%)

## Story — versão 1

## Sugestão inicial gerada a partir da demanda

Você pode aprovar esta versão diretamente ou usar o chat para solicitar ajustes.

SUGESTÃO PRELIMINAR DE HISTÓRIA DE USUÁRIO

Entendimento Inicial (factual):
O usuário solicita a criação de um novo contexto para o veículo Fiat Argo no sistema CarHelper, com o objetivo de responder perguntas sobre esse automóvel. Solicita, também, a obtenção do manual em PDF desse modelo e a submissão deste à aplicação.  
Atualmente, conforme análise dos trechos recuperados do repositório (Fontes 1 a 3), o sistema está estruturado para responder exclusivamente sobre Ford KA e Fiat Mobi, através de agentes especializados e um agente generalista, todos utilizando informações de documentos previamente carregados e sem permitir conhecimento prévio ou suposição.  
Não há, nos trechos recuperados, referência à existência de agentes, ferramentas ou contexto para o Fiat Argo. Todas as referências recuperadas citam Ford KA e Fiat Mobi. Não há informação, tampouco, sobre o procedimento técnico de upload de novos documentos, tampouco o formato esperado dos arquivos/documentos.

História de Usuário (PRELIMINAR)  
Como usuário do sistema CarHelper  
Quero que o sistema ofereça um contexto de atendimento específico para o veículo Fiat Argo, com base em manual/documentação oficial  
Para que dúvidas técnicas relacionadas ao Fiat Argo possam ser respondidas de modo semelhante ao que já ocorre com Ford KA e Fiat Mobi

Critérios de Aceitação (Dado / Quando / Então)
- Dado que existe o contexto do Fiat Argo cadastrado na aplicação  
  Quando um usuário faz uma pergunta técnica sobre o Fiat Argo  
  Então o sistema identifica corretamente a intenção e fornece uma resposta baseada nos documentos do Fiat Argo, conforme já acontece com Ford KA e Fiat Mobi.

- Dado que o manual do Fiat Argo foi carregado na aplicação  
  Quando o agente precisa extrair informações para responder uma pergunta relacionada ao Fiat Argo  
  Então a busca e a extração devem se limitar exclusivamente ao conteúdo recuperado do manual/documento relevante, sem uso de conhecimento externo ou prévio.

Dúvidas Abertas (necessárias para evolução da história)
1. Qual o processo técnico, no CarHelper, para adicionar um novo veículo (contexto) e os respectivos documentos? Existe um modelo de agente ou ferramenta a ser seguido?
2. O formato do manual do Fiat Argo (PDF) é compatível automaticamente com o sistema ou requer conversão/preparo especial para ingestão?
3. Há necessidade de alterar regras existentes (ex: do agente recepcionista ou generalista) para contemplar o novo veículo ou criar agentes específicos?
4. O sistema deve oferecer respostas simultâneas comparando o Fiat Argo a outros veículos, ou o atendimento será individual por contexto?
5. Quem é responsável pela obtenção e validação do manual do Fiat Argo — haverá integração automática ou upload manual do documento?

Referências do repositório consultadas

- [Fonte 1: CarHelper/agents/recepcionist/agent.py] — Estrutura do agente recepcionista, regras A/B/C para classificação e roteamento das perguntas, listando Ford KA e Fiat Mobi como veículos suportados.
- [Fonte 2: CarHelper/agents/specialists/generalist/agent.py] — Agente generalista lida com perguntas técnicas sobre ambos os modelos suportados.
- [Fonte 3: CarHelper/agents/specialists/agent.py] — Define a obrigatoriedade do uso exclusivo de informação extraída de documentos, sem suposições ou conhecimento prévio.
- [Fonte 4: CarHelper/repository-profile] — Confirmação do branch, linguagem (Python) e contexto do repositório sincronizado até o momento.

Limitações Observadas
- Não há informações técnicas sobre como adicionar novos contextos ou documentos ao sistema.
- Falta referência a algum agente, rotina, classe, toolset ou processo já existente para lidar com veículos além de Ford KA e Fiat Mobi.
- NÃO se deve inventar regras, nomes de arquivos ou de API que não constem no material recuperado.

Esta história é preliminar por não termos acesso a detalhes operacionais de cadastro/ingestão de novos veículos e documentos no CarHelper.  
É recomendada a obtenção de maiores esclarecimentos e/ou dos arquivos onde se processa a entrada de novos documentos/veículos para avançar com uma proposta mais detalhada.

## Refinement — versão 1

## Primeira versão gerada automaticamente

Este resultado usa como entrada o output aprovado da etapa anterior. Você pode aprová-lo diretamente ou usar o chat para solicitar ajustes.

## Refinamento da História de Usuário — Novo contexto FiatArgo no CarHelper

### 1. Requisitos Funcionais

**RF1.** O sistema deve permitir o cadastro de um novo contexto para o veículo Fiat Argo.
  - *DÚVIDA*: O procedimento de cadastro de novo contexto e manual/documentação no CarHelper não foi definido nem localizado nos arquivos sincronizados.

**RF2.** O sistema deve permitir a ingestão do manual/documentação oficial do Fiat Argo em formato PDF.
  - *DÚVIDA*: Não há garantias sobre compatibilidade automática de PDFs ou necessidade de conversão/pre-processamento.

**RF3.** O sistema deve responder perguntas técnicas sobre o Fiat Argo utilizando exclusivamente informações do manual/documentação carregada.
  - Comportamento esperado é idêntico ao já implementado para Ford KA e Fiat Mobi.

**RF4.** O sistema deve identificar automaticamente que uma pergunta está relacionada ao Fiat Argo e direcionar para o novo contexto/agente correspondente.
  - *DÚVIDA*: Não está claro se é necessário criar um novo agente especializado (ex: `especialista_fiatargo`) ou adaptar agentes existentes.

**RF5.** As respostas devem ser baseadas estritamente nas informações recuperadas do manual/documentação, vedado o uso de conhecimento prévio ou suposições.
  - Segue padrão de comportamento dos agentes já existentes (Ford KA, Fiat Mobi, Generalista).

**RF6.** O CarHelper deve manter o comportamento dos agentes especialistas existentes para Ford KA e Fiat Mobi.

**RF7.** O CarHelper deve garantir o funcionamento simultâneo de múltiplos contextos/veículos sem afetar os veículos já suportados.

---

### 2. Requisitos Não Funcionais

**RNF1.** O novo contexto Fiat Argo deve ser processado sem impactar a performance ou disponibilidade dos contextos já existentes.
**RNF2.** Não deve haver aumento indevido de latência na resposta para perguntas relativas a qualquer veículo suportado.
**RNF3.** A compatibilidade do formato do manual do Fiat Argo (PDF) deve ser verificada ou documentada.
  - *DÚVIDA*: Falta especificação do pipeline de ingestão de documentos.
**RNF4.** O sistema deve se manter aderente às práticas de não utilizar conhecimento externo, conforme já garantido nos outros contextos.

---

### 3. Cenários BDD

#### Cenário 1 — Resposta a pergunta técnica sobre Fiat Argo
**Dado** que o manual do Fiat Argo foi carregado e o contexto está ativo  
**Quando** um usuário faz uma pergunta sobre manutenção do Fiat Argo  
**Então** o sistema identifica a intenção, direciona ao contexto Fiat Argo, e responde exclusivamente com base no manual/documentação do Fiat Argo

#### Cenário 2 — Garantia de isolamento de contexto
**Dado** que existe o manual do Fiat Argo e de outros veículos no sistema  
**Quando** uma pergunta faz referência explícita ao Fiat Argo  
**Então** o sistema utiliza apenas o documento do Fiat Argo, sem buscar ou misturar informações de outros veículos

#### Cenário 3 — Pergunta fora do escopo
**Dado** que uma pergunta não é sobre Ford KA, Fiat Mobi ou Fiat Argo  
**Quando** usuário envia essa pergunta ao CarHelper  
**Então** o sistema informa que só responde dúvidas técnicas sobre Ford KA, Fiat Mobi e Fiat Argo (caso o contexto tenha sido aprovado para inclusão na lógica do agente recepcionista)

---

### 4. Dependências

- Estrutura para definição de novos agentes na pasta `CarHelper/agents/specialists/` (exemplos: `fordka/agent.py`, `mobi/agent.py`).  
- Registro do manual/documentação do Fiat Argo — processo e local de ingestão indisponíveis nos trechos sincronizados (*DÚVIDA*).
- Regras de roteamento do agente recepcionista em `CarHelper/agents/recepcionist/agent.py` (que atualmente só cita Ford KA e Fiat Mobi — precisa ser adaptado).
- Possível dependência da biblioteca `google.adk.tools` e flows internos.

---

### 5. Impactos de Dados

- Inclusão de novos dados (conteúdo manual do Fiat Argo) nas fontes consultadas pelos agentes.
- Não há detalhes sobre modelos de armazenamento, versionamento, ou estrutura de documentos nos arquivos recuperados (*DÚVIDA*).
- Risco de contaminação do contexto ou “leak” de informações entre veículos se isolamento não for garantido na implementação do novo agente/contexto.

---

### 6. APIs/Componentes Impactados

#### Confirmados (com base nos caminhos recuperados):

- `CarHelper/agents/recepcionist/agent.py`
- `CarHelper/agents/specialists/` (estrutura de agentes: `generalist/`, `mobi/`, `fordka/`). 
  - Presume-se que *deverá* existir um `fiatargo/agent.py` seguindo padrão dos demais.
- Ferramentas de base `google.adk.tools` para agentes e ingestão.

#### Não localizados ou insuficientemente documentados:

- API/rotina de upload, ingestão ou processamento de documento/manual.
- Estrutura de configuração dinâmica ou registro de novos veículos.

---

### 7. Riscos

- **Risco de sincronização**: Falta referência ao mecanismo de ingestão/registro de contexto/veículo/documento. Implementação pode colidir com padrões ocultos ou processos específicos do sistema.
- **Risco de misturas de contexto**: Sem clara separação de buscas ou agentes, é possível que respostas tragam outros veículos se a regra do agente não for adaptada.
- **Risco de incompatibilidade com PDF**: Manual pode requerer transformação/conversão.
- **Risco de impacto nas regras do recepcionista**: O agente precisa ter a regra adaptada para o Fiat Argo; ausência pode impedir roteamento correto.
- **Risco de manutenção**: Padronização de novos veículos/agentes pode divergir sem um modelo centralizado.

---

### 8. Observabilidade

- Necessário registrar logs detalhados de perguntas sobre Fiat Argo e seu roteamento/contexto para validação da implementação.
- Monitorar se consultas e respostas de agentes estão, de fato, isoladas ao contexto/veículo solicitado.
- *DÚVIDA*: Não há detalhe sobre infraestrutura de logs, tracing ou ferramentas de observabilidade presentes no sistema.

---

### 9. Segurança

- Controle de acesso à área de upload/cadastro de novos documentos (se aplicável).
- Garantia de que o conteúdo do manual não será extraído ou exposto de forma indevida.
- *DÚVIDA*: Não há detalhes sobre práticas de autenticação/autorização, upload ou sanização de arquivos.

---

### 10. Acessibilidade

- O novo fluxo de perguntas e respostas para o Fiat Argo deve manter as mesmas características de acessibilidade presentes na interface para Ford KA e Fiat Mobi.
- *DÚVIDA*: Não há informações nos trechos sincronizados sobre práticas ou ferramentas de acessibilidade aplicadas ao frontend ou chatbot.

---

### 11. Tarefas Técnicas

**1. Analisar e documentar o processo atual de inclusão de novos veículos/contextos e ingestão de documentos (manual).**
- *DÚVIDA*: Não identificado nos arquivos sincronizados.

**2. Criar agente especializado para Fiat Argo, espelhando o padrão em:**
   - `CarHelper/agents/specialists/mobi/agent.py`
   - `CarHelper/agents/specialists/fordka/agent.py`

**3. Adaptar regras do agente recepcionista em `CarHelper/agents/recepcionist/agent.py`**
- Incluir menção ao Fiat Argo nas regras e roteamento de perguntas.

**4. Adaptar, caso necessário, o agente generalista para permitir futuras comparações ou respostas multi-contexto (se for diretriz do produto).**
- *DÚVIDA*: A necessidade de comparação múltipla não foi confirmada.

**5. Garantir a ingestão e indexação do manual do Fiat Argo para uso pelos agentes.**
- Validar formato do arquivo e processo necessário.

**6. Realizar testes de pergunta/resposta end-to-end assegurando isolamento de contexto e aderência à política de não-uso de conhecimento externo.**

**7. Atualizar documentação do projeto sobre como adicionar novos contextos e documentos.**

---

### 12. Lacunas e Dúvidas

- Modelo, rotina e API para upload/ingestão de novos documentos não identificado.  
- Não está claro se a comparação entre múltiplos veículos será suportada ou se o agente generalista terá que ser adaptado.
- Não se sabe como (ou se) o frontend ou backend sinaliza visualmente ou via interface que um novo contexto foi adicionado.
- Não há informações sobre versionamento, fallback ou rollback de fontes/documentos.
- A compatibilidade com PDF e pipeline de extração de conteúdo são totalmente desconhecidos.

---

### 13. Caminhos Considerados

- Replicar o padrão de agentes especialistas identificado para Ford KA e Fiat Mobi para um novo agente Fiat Argo — suportado pelos arquivos sincronizados.
- Adaptar o agente recepcionista, visto que as regras explicitam a necessidade de reconhecer nomes dos veículos suportados e seguir regras de roteamento específicas.
- Não contemplar rotinas de upload/processamento de documentos, visto que não há qualquer referência a isso nas fontes.

---

### 14. Considerações Finais

O avanço técnico desta demanda depende da disponibilização de:
- Processos/rotinas para ingestão de novos documentos e cadastro de contextos.
- Esclarecimentos sobre o tratamento de novas features no pipeline de NLP, ingestão e interface.
- Expansão dos arquivos sincronizados para as rotinas e estruturas de dados/documentos do CarHelper.

Enquanto o padrão identificado aponta para fácil replicação da estrutura de agentes, quaisquer decisões sobre ingestão, arquivo, API ou UI não devem ser tomadas sem evidência concreta nos arquivos recuperados ou sinalização clara sobre o produto.

---

**Resumo**
O refinamento acima traça um roteiro prático mínimo, apresenta todos os pontos dependentes de informação estrutural/fundacional ausente nos arquivos consultados, e garante alinhamento com os padrões do CarHelper conforme material disponível. Para avanço estrutural, novas sincronizações são obrigatórias.

## Proposta do Agente de Desenvolvimento

## Primeira versão gerada automaticamente

Este resultado usa como entrada o output aprovado da etapa anterior. Você pode aprová-lo diretamente ou usar o chat para solicitar ajustes.

## Proposta de Desenvolvimento — Novo Contexto Fiat Argo no CarHelper

**Repositório-alvo:** CarHelper  
**Branch base:** main  
**Linguagem principal:** Python  
**Estrutura consultada:**  
- `CarHelper/agents/specialists/mobi/agent.py`
- `CarHelper/agents/specialists/fordka/agent.py`
- `CarHelper/agents/recepcionist/agent.py`
- **Não sincronizado**: pipeline de ingestão de documentos, configuração global de agentes/contextos.

---

### Plano Técnico de Implementação

#### 1. Criação de agente especializado Fiat Argo

- Replicar, com as devidas adaptações de veículo, o padrão dos agentes para Ford KA e Fiat Mobi.
- Arquivo novo:  
  - `CarHelper/agents/specialists/fiatargo/agent.py`

#### 2. Adaptação do agente recepcionista

- Alteração necessária para incluir o Fiat Argo nas regras do recepcionista:
  - Roteamento de perguntas técnicas explícitas ao Fiat Argo para o novo agente.
  - Atualização da lista de veículos suportados e instruções.

- Arquivo afetado:
  - `CarHelper/agents/recepcionist/agent.py`

#### 3. Processo de ingestão do manual/documentação

- **Dúvida não resolvida**: Não foram localizados trechos, APIs, comandos ou pipelines referentes à ingestão de PDFs/documentos. Impossível propor tecnicamente qualquer patch, script ou comando sem evidência concreta.
- Recomenda-se solicitar sincronização/expansão de arquivos e esclarecimento sobre:
  - Local padrão de armazenamento do manual do veículo.
  - Pipeline de ingestão/indexação de PDFs.

#### 4. Testes e validação

- Estruturar cenários de teste end-to-end:
  - Criação de perguntas técnicas sobre o Fiat Argo.
  - Garante que as respostas sejam baseadas apenas no contexto/documentação do Fiat Argo.
  - Garante que não há leak de contexto dos demais veículos.

- Precisa de:
  - Permissão para criar/adaptar scripts de teste, normalmente em pasta como `tests/`.

---

### Arquivos Prováveis a Criar ou Alterar

| Caminho                                        | Tipo         | Ação         |
|------------------------------------------------|--------------|--------------|
| CarHelper/agents/specialists/fiatargo/agent.py | Python       | Novo         |
| CarHelper/agents/recepcionist/agent.py         | Python       | Alterado     |

---

#### Sugestão de novo arquivo: CarHelper/agents/specialists/fiatargo/agent.py

```python
# app_agents.py
from __future__ import annotations
from google.adk.tools.agent_tool import AgentTool
from agents.specialists.agent import SpecialistAgent

fiatArgo = AgentTool(agent = SpecialistAgent(
        name="especialista_fiatargo",
        description=(
            "Especialista técnico no Fiat Argo. "
            "Use este agente para perguntas sobre especificações, manutenção, revisões, peças e uso do Fiat Argo."
            "Considere os documentos recuperados da base de dados do Fiat Argo para responder às perguntas, sem adicionar conhecimento prévio ou suposições."
            "Utilize as ferramentas de consulta para extrair informações diretamente dos documentos do Fiat Argo, mesmo que a pergunta pareça simples ou direta. Nunca responda com base em conhecimento prévio ou suposições."
            "Crie a query para as ferramentas de consulta de forma a extrair informações relevantes dos documentos do Fiat Argo, mesmo que a pergunta pareça simples ou direta. Nunca responda com base em conhecimento prévio ou suposições."
        )
    )
)
```
> **Obs.:** O padrão do arquivo segue o modelo de Ford KA e Fiat Mobi.

---

#### Alteração em CarHelper/agents/recepcionist/agent.py

**Onde atualizar:**
- Incluir menção ao Fiat Argo nas instruções, regras e roteamento.

**Sugestão de alteração (parcial, substituição dos veículos nos trechos relevantes):**

Dentro do campo `instruction`, adaptar:

Antes:
```python
"Você é o recepcionista do sistema de dúvidas sobre Ford KA e Fiat Mobi.\n\n"
...
"  - Use 'especialista_fordka' para perguntas claramente relacionadas ao Ford KA.\n"
"  - Use 'especialista_fiatmobi' para perguntas claramente relacionadas ao Fiat Mobi.\n"
"  - Use 'especialista_generalista' para perguntas técnicas mais genéricas sobre carros, trazendo informações do Ford KA e Fiat Mobi.\n"
```

Adaptar para incluir Fiat Argo:

```python
"Você é o recepcionista do sistema de dúvidas sobre Ford KA, Fiat Mobi e Fiat Argo.\n\n"
...
"  - Use 'especialista_fordka' para perguntas claramente relacionadas ao Ford KA.\n"
"  - Use 'especialista_fiatmobi' para perguntas claramente relacionadas ao Fiat Mobi.\n"
"  - Use 'especialista_fiatargo' para perguntas claramente relacionadas ao Fiat Argo.\n"
"  - Use 'especialista_generalista' para perguntas técnicas mais genéricas sobre carros, trazendo informações do Ford KA, Fiat Mobi e Fiat Argo.\n"
```

> **Nota:** O código específico da classe e roteamento depende de como é feito o registro da tool `especialista_fiatargo` na lista de ferramentas, que não está detalhado nos trechos recuperados.

---

### Testes Sugeridos

**Cenários:**
- Pergunta técnica relacionada explicitamente ao Fiat Argo: deve ser roteada ao novo agente e respondida exclusivamente com base no manual do Fiat Argo.
- Pergunta técnica sobre Ford KA ou Fiat Mobi: isolamento e sem impacto do novo contexto.
- Pergunta técnica genérica (ex: "Como trocar pneu?"): roteamento para o agente generalista, incluindo informações do Fiat Argo, Ford KA e Fiat Mobi.
- Pergunta fora do escopo dos veículos: resposta educada com a lista atualizada de veículos.

**Teste manual:**  
- Enviar perguntas por interface ou script:  
  - "Como faço a revisão do Fiat Argo?" → resposta baseada no manual do Fiat Argo.
  - "Como faço a revisão do Ford KA?" → resposta baseada no manual do Ford KA.
  - "Como trocar pneu?" → comparar outputs com informações dos três veículos.

**Teste automatizado:**  
- Não há referência a um framework de teste na estrutura sincronizada. Recomenda-se futura criação de scripts em pasta como `tests/` caso haja suporte.

---

### Comandos de Validação Sugeridos

**Validação manual:**  
- Revisar resposta do chatbot/flow para perguntas sobre Fiat Argo.
- Inspecionar logs de roteamento, se disponíveis.

**Validação técnica:**  
- Verificar se o agente Fiat Argo está registrado no sistema (depende de registro na configuração de ferramentas, não sincronizado).

---

### Riscos

- **Processo de ingestão/documentação ausente:** Se não houver ingestão do manual Fiat Argo, o agente será criado mas não terá fonte de dados.
- **Registro de ferramenta:** Falta evidência sobre como as tools são registradas/instanciadas; risco de patch parcial ou incompatível.
- **Isolamento de contexto:** Se o sistema não isola corretamente, pode misturar informações entre veículos.
- **Compatibilidade do PDF:** Manual pode requerer processamento não documentado.

---

### Proposta de Rollback

- Para rollback, basta remover o arquivo `fiatargo/agent.py` e reverter as alterações em `recepcionist/agent.py`.
- Não há dependências transitivas documentadas.

---

### Solicitações de Contexto

- **Manual/documento:** Solicita localização ou informação sobre pipeline de ingestão.
- **Registro de tools/agents:** Solicita evidência sobre como o novo agente é registrado ou disponibilizado ao recepcionista.
- **Testes/scripts:** Solicita acesso a framework/pastas de teste caso haja para testagem automatizada.

---

## Resumo

A proposta avança apenas com a estrutura e adaptação dos agentes, alinhada ao modelo do sistema. Não há ação sobre ingestão de documentos/PDFs ou infraestrutura, pois tais processos não foram sincronizados. Recomenda-se sincronizar arquivos relacionados ao pipeline de ingestão, registro de tools, scripts de teste e documentação de contexto para garantir a efetividade da implementação.

**Aguardando revisão e aprovação para criação da branch de desenvolvimento e Draft PR.**

## Comentário da aprovação humana

Aprovado sem comentário adicional.

## Guardrails

- O PR é criado como **Draft**.
- Merge e deploy continuam sendo decisões humanas.
- O conteúdo deve ser revisado antes de qualquer implementação ou incorporação.
