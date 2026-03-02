# Agent Composer Masterclass: Complete Guide

## Table of Contents
1. [Core Concepts](#1-core-concepts)
2. [Architecture & Execution Model](#2-architecture--execution-model)
3. [All Node Types](#3-all-node-types)
4. [YAML Syntax Deep Dive](#4-yaml-syntax-deep-dive)
5. [Agentic Workflows](#5-agentic-workflows)
6. [Conditional Logic & Routing](#6-conditional-logic--routing)
7. [Subgraphs & Reusability](#7-subgraphs--reusability)
8. [External Integrations](#8-external-integrations)
9. [Retrieval Configuration](#9-retrieval-configuration)
10. [Generation Configuration](#10-generation-configuration)
11. [UI Streaming](#11-ui-streaming)
12. [Evaluation with LMUnit](#12-evaluation-with-lmunit)
13. [Best Practices & Patterns](#13-best-practices--patterns)

---

## 1. Core Concepts

### What is Agent Composer?

Agent Composer is a framework for building **custom RAG agents as computational graphs**. Instead of using a fixed pipeline, you compose workflows from pre-built nodes that handle search, generation, transformation, and logic.

### Two Ways to Build

| Method | Description | Access |
|--------|-------------|--------|
| **YAML Configuration** | Define workflows in structured YAML files | Enterprise |
| **Visual Workflow Builder** | Drag-and-drop GUI that generates YAML | Enterprise |
| **Templates** | Pre-built workflows (Basic, Agentic Search) | Self-serve |

### Key Terminology

| Term | Definition |
|------|------------|
| **Node** | A single operation (search, generate, transform, etc.) |
| **Input Mapping** | How data flows between nodes (`node#output`) |
| **Config** | Parameters passed to a node's constructor |
| **Subgraph** | A reusable workflow that can be nested |
| **Conditional Node** | Branches execution based on runtime conditions |
| **UI Stream Types** | What data streams to the user during execution |

---

## 2. Architecture & Execution Model

### How It Works

Agent Composer executes workflows as **Directed Acyclic Graphs (DAGs)**:

```
[__inputs__] → [Node A] → [Node B] → [Node C] → [__outputs__]
                  ↓
              [Node D] → [Node E] ↗
```

1. Query enters through `__inputs__`
2. Nodes execute based on dependency order
3. Each node receives inputs from mapped sources
4. Each node produces outputs for downstream nodes
5. Final results collected at `__outputs__`

### Execution Rules

- **All inputs must be wired** — every node input needs a source
- **Outputs must be JSON-serializable** — primitives, lists, dicts
- **Nodes execute when inputs are ready** — parallel when possible
- **No cycles allowed** — DAG only

---

## 3. All Node Types

### Search & Retrieval Nodes

| Node | Purpose | Key Config |
|------|---------|------------|
| `SearchUnstructuredDataStep` | Search your datastores | `top_k`, `lexical_alpha`, `semantic_alpha`, `reranker` |
| `MetadataSearchStep` | Multi-hop reasoning over metadata | Enables "find papers cited by X" |
| `WebSearchStep` | Live web search | `top_k` |
| `QueryStructuredDatastoreStep` | Query structured databases | SQL-like queries |

### Retrieval Processing Nodes

| Node | Purpose | Key Config |
|------|---------|------------|
| `RerankRetrievalsStep` | Reorder results by relevance | `rerank_top_k`, `rerank_instructions`, `reranker_score_filter_threshold` |
| `FilterRetrievalsStep` | Remove low-quality chunks | `template_filter`, `filter_retrievals` |
| `ReformulateQueryStep` | Expand/rewrite queries | `instructions` |

### Generation Nodes

| Node | Purpose | Key Config |
|------|---------|------------|
| `ResponseGenerationStep` | Generate response with attributions | `temperature`, `enable_groundedness_check` |
| `GenerateFromResearchStep` | Generate from agentic research output | `model`, `temperature`, `system_prompt` |
| `LanguageModelStep` | Generic LLM prompting | `prompt`, `model`, `temperature` |

### Agentic Nodes

| Node | Purpose | Key Config |
|------|---------|------------|
| `AgenticResearchStep` | Multi-turn agent loop with tools | `agent_config`, `tools_config`, `num_turns` |
| `CreateMessageHistoryStep` | Initialize conversation context | Required before `AgenticResearchStep` |

### Transformation Nodes

| Node | Purpose | Key Config |
|------|---------|------------|
| `JSONCreatorStep` | Create JSON with variable substitution | Template with `$variable` syntax |
| `GetMemberStep` | Extract field from object | `field_name` |
| `SetMemberStep` | Set field in object | `field_name`, `value` |
| `WrapStep` | Wrap value in container | |
| `MergeStep` | Merge multiple inputs | |

### Conditional & Logic Nodes

| Node | Purpose | Key Config |
|------|---------|------------|
| `ConditionalStep` | If/else branching | `variable`, `branches` |

### Integration Nodes

| Node | Purpose | Key Config |
|------|---------|------------|
| `WebhookStep` | Call external REST APIs | `webhook_url`, `method`, `auth_token` |
| `MCPClientStep` | Connect to MCP servers | `server_url`, `tool_name`, `tool_args` |
| `CodeExecutionStep` | Run Python code (sandboxed) | `model` (uses Gemini) |
| `SalesforceSOSLStep` | Query Salesforce | `sosl_query` |
| `ContextualAgentStep` | Query another Contextual agent | `agent_id` |

---

## 4. YAML Syntax Deep Dive

### Root Structure

```yaml
inputs:
  query: str                    # Main graph only accepts query: str

outputs:
  response: str                 # Must be JSON-serializable
  attributions: object
  score: float

nodes:
  # Your nodes here

__outputs__:                    # Required - maps final outputs
  type: output
  ui_output: true               # Required if multiple outputs
  input_mapping:
    response: generate#response
    attributions: generate#attribution_result
```

### Node Structure

```yaml
node_name:
  type: StepType                # Required
  config:                       # Constructor parameters
    param1: value1
    param2: value2
  config_overrides:             # Dynamic runtime values
    param3: other_node#output
  input_mapping:                # Data sources
    input1: __inputs__#query
    input2: prev_node#output
  ui_stream_types:              # What to stream to UI
    generation: true
    retrievals: true
```

### Input Mapping Syntax

| Pattern | Meaning |
|---------|---------|
| `__inputs__#query` | Graph-level input |
| `node_name#output_key` | Output from another node |
| `subgraph_name#output` | Output from subgraph |

### Config Overrides

Pull values dynamically at runtime:

```yaml
search:
  type: SearchUnstructuredDataStep
  config:
    semantic_alpha: 0.8          # Static
  config_overrides:
    top_k: dynamic_k#output      # Dynamic from another node
  input_mapping:
    query: __inputs__#query
```

---

## 5. Agentic Workflows

### The Agentic Pattern

```yaml
# Phase 1: Initialize
init_history:
  type: CreateMessageHistoryStep
  input_mapping:
    query: __inputs__#query

# Phase 2: Research (agent loops with tools)
research:
  type: AgenticResearchStep
  config:
    agent_config:
      model: "claude-sonnet-4-5-20250514"
      num_turns: 10                    # Max iterations
      enable_parallel_tool_calls: true
      identity_prompt: "You are..."
      research_guidelines_prompt: "Search strategy..."
      output_guidelines_prompt: "Format output as..."
    tools_config:
      - name: tool_name
        description: "When to use this tool"
        step_config:
          type: SearchUnstructuredDataStep
          config: { ... }
  input_mapping:
    message_history: init_history#message_history

# Phase 3: Generate from research
generate:
  type: GenerateFromResearchStep
  config:
    model: "claude-sonnet-4-5-20250514"
    temperature: 0.3
  input_mapping:
    research_output: research#output
    query: __inputs__#query
```

### Agent Config Options

| Option | Description | Default |
|--------|-------------|---------|
| `model` | LLM for reasoning | claude-sonnet-4-5 |
| `num_turns` | Max tool-use iterations | 10 |
| `enable_parallel_tool_calls` | Call multiple tools at once | false |
| `identity_prompt` | Agent persona/role | |
| `research_guidelines_prompt` | How to research | |
| `output_guidelines_prompt` | How to format findings | |

### Tool Definition

**Simple tool (single step):**
```yaml
tools_config:
  - name: search_docs
    description: "Search internal documentation for..."
    step_config:
      type: SearchUnstructuredDataStep
      config:
        top_k: 50
        semantic_alpha: 0.9
```

**Complex tool (subgraph):**
```yaml
tools_config:
  - name: deep_search
    description: "Comprehensive search with filtering"
    graph_config:
      inputs:
        query: str
      outputs:
        results: object
      nodes:
        search:
          type: SearchUnstructuredDataStep
          ...
        filter:
          type: FilterRetrievalsStep
          ...
        __outputs__:
          type: output
          input_mapping:
            results: filter#retrievals
```

---

## 6. Conditional Logic & Routing

### ConditionalStep Structure

```yaml
classify:
  type: LanguageModelStep
  config:
    prompt: "Classify this query as 'factual' or 'creative': {query}"
  input_mapping:
    query: __inputs__#query

route:
  type: ConditionalStep
  config:
    variable: classification       # Field to evaluate
    branches:
      - condition: "== factual"
        executable:
          type: subgraph:factual_handler
        input_mapping:
          query: __inputs__#query
        output_mapping:
          response: response
      - condition: "== creative"
        executable:
          type: subgraph:creative_handler
        input_mapping:
          query: __inputs__#query
        output_mapping:
          response: response
  input_mapping:
    classification: classify#output
```

### Supported Operators

| Operator | Example |
|----------|---------|
| `==` | `"== factual"` |
| `!=` | `"!= spam"` |
| `>` | `"> 0.5"` |
| `>=` | `">= 0.8"` |
| `<` | `"< 10"` |
| `<=` | `"<= 100"` |

### Routing Patterns

**Query Type Routing:**
```
Query → Classify → Route → [Technical Handler | General Handler | Creative Handler]
```

**Confidence-Based Routing:**
```
Query → Search → Score → Route → [High Confidence Path | Low Confidence Path]
```

**Source-Specific Routing:**
```
Query → Detect Intent → Route → [arXiv Search | Reddit Search | Both]
```

---

## 7. Subgraphs & Reusability

### Defining a Subgraph

```yaml
# Define at root level alongside main graph
document_retrieval:              # Subgraph name
  inputs:
    query: str
    top_k: int
  outputs:
    retrievals: object
  nodes:
    search:
      type: SearchUnstructuredDataStep
      config:
        semantic_alpha: 0.9
      config_overrides:
        top_k: __inputs__#top_k
      input_mapping:
        query: __inputs__#query

    rerank:
      type: RerankRetrievalsStep
      config:
        rerank_top_k: 10
      input_mapping:
        query: __inputs__#query
        retrievals: search#retrievals

    __outputs__:
      type: output
      input_mapping:
        retrievals: rerank#retrievals
```

### Using a Subgraph

```yaml
nodes:
  retriever:
    type: subgraph:document_retrieval    # Prefix with subgraph:
    input_mapping:
      query: __inputs__#query
      top_k: 50
```

### Subgraph Benefits

- **Reusability** — Define once, use multiple times
- **Encapsulation** — Internal nodes isolated from parent
- **Nesting** — Subgraphs can contain subgraphs
- **As Tools** — Attach to `AgenticResearchStep` as callable tools

---

## 8. External Integrations

### Webhooks (REST APIs)

```yaml
call_api:
  type: WebhookStep
  config:
    webhook_url: "https://api.example.com/endpoint"
    method: POST                    # GET, POST, PUT, DELETE, PATCH
    auth_token: "${API_TOKEN}"      # Bearer token
    timeout: 30                     # Seconds
    retries: 2
    static_headers:
      Content-Type: "application/json"
      X-Custom-Header: "value"
  input_mapping:
    context_data: prev_node#output  # Becomes request body
```

### MCP Integration

```yaml
crm_lookup:
  type: MCPClientStep
  config:
    server_url: "https://mcp.example.com"
    tool_name: "lookup_customer"
    tool_args:
      customer_id: "$customer_id"   # Variable substitution
      fields: ["name", "email"]
    transport_type: http
    connection_timeout: 30
    auth_headers:
      Authorization: "Bearer ${CRM_TOKEN}"
  input_mapping:
    customer_id: extract#customer_id
```

### Code Execution

```yaml
calculate:
  type: CodeExecutionStep
  config:
    model: "gemini-2.5-flash"       # Uses Gemini
  input_mapping:
    query: __inputs__#query         # Natural language request
```

**Security:**
- No network access
- No file system access
- Limited execution time and memory

### Salesforce Integration

```yaml
salesforce:
  type: SalesforceSOSLStep
  config:
    sosl_query: "FIND {$search_term} IN ALL FIELDS RETURNING Account, Contact"
    result_limit: 250
    timeout: 30
    # OAuth auth
    client_id: "${SF_CLIENT_ID}"
    client_secret: "${SF_CLIENT_SECRET}"
```

---

## 9. Retrieval Configuration

### Search Parameters

| Parameter | Range | Default | Description |
|-----------|-------|---------|-------------|
| `top_k` / `top_k_retrieved_chunks` | 1-200 | 100 | Max chunks to retrieve |
| `semantic_alpha` | 0-1 | 0.9 | Semantic search weight |
| `lexical_alpha` | 0-1 | 0.1 | Keyword search weight (must sum to 1 with semantic) |
| `enable_query_expansion` | bool | false | Rewrite query with terminology |
| `enable_query_decomposition` | bool | false | Break complex queries into sub-queries |

### Reranking Parameters

| Parameter | Range | Default | Description |
|-----------|-------|---------|-------------|
| `reranker` | string | ctxl-rerank-v2-instruct-multilingual-FP8 | Reranker model |
| `rerank_top_k` / `top_k_reranked_chunks` | 1-200 | — | Max chunks after reranking |
| `reranker_score_filter_threshold` | 0-1 | — | Min score to keep |
| `rerank_instructions` | string | — | Natural language ranking preferences |

### Rerank Instructions Examples

```yaml
rerank_instructions: |
  Prioritize:
  - Recent content (last 6 months) for fast-moving topics
  - High-citation papers for established techniques
  - Highly-upvoted posts for practical advice

  Deprioritize:
  - Surveys and literature reviews
  - Low-effort posts without technical detail
  - Outdated information (>2 years old)
```

### Chunking Strategies

| Strategy | Best For | Description |
|----------|----------|-------------|
| **Hierarchy Depth** (default) | Academic papers, manuals | Uses document structure (sections, subsections) |
| **Hierarchy Heading** | Contracts, chat logs | Segments by headings, ignores depth |
| **Static Length** | Baseline testing | Fixed token intervals |
| **Page Level** | Slide decks, summaries | Each page = one chunk |

---

## 10. Generation Configuration

### Generation Parameters

| Parameter | Range | Default | Description |
|-----------|-------|---------|-------------|
| `temperature` | 0-1 | — | Randomness (lower = more deterministic) |
| `top_p` | 0-1 | — | Nucleus sampling |
| `max_new_tokens` | — | — | Max output length |
| `frequency_penalty` | — | — | Reduce repetition |
| `random_seed` | — | — | Reproducibility |

### System Prompt

```yaml
generate:
  type: ResponseGenerationStep
  config:
    system_prompt: |
      You are a technical research assistant.

      Guidelines:
      - Only use information from provided documentation
      - Use exact terminology from sources
      - Keep answers concise and relevant
      - Use markdown for lists, tables, code
      - Answer directly, then stop
      - If information is missing, say so clearly
```

### Groundedness & Attribution

```yaml
generate:
  type: ResponseGenerationStep
  config:
    enable_groundedness_check: true   # Score how grounded in sources
    enable_attribution: true          # Extract citation metadata
```

**Outputs:**
- `response` — Generated text
- `attribution_result` — Citation metadata
- `groundedness_scores` — How well grounded in sources

---

## 11. UI Streaming

### Stream Types

| Type | Description |
|------|-------------|
| `RETRIEVALS` | Search results and citations |
| `GENERATION` | LLM response as it generates |
| `QUERY_REFORMULATION` | Query transformations |
| `ATTRIBUTION` | Citation metadata |
| `GROUNDEDNESS` | Grounding scores |

### Configuration

```yaml
generate:
  type: ResponseGenerationStep
  ui_stream_types:
    generation: true      # Stream response text
    retrievals: true      # Show retrieved chunks
    attribution: true     # Show citations
    groundedness: true    # Show grounding scores
```

### Multiple Outputs

When your graph has multiple outputs, designate which one streams to UI:

```yaml
outputs:
  response: str
  debug_info: object

__outputs__:
  type: output
  ui_output: true         # Required
  input_mapping:
    response: generate#response       # This one streams
    debug_info: debug#info
```

---

## 12. Evaluation with LMUnit

### What is LMUnit?

LMUnit evaluates LLM outputs through **natural language unit tests**. Instead of generic metrics, you define specific testable criteria.

### Writing Unit Tests

**Good tests are:**
- Specific (one criterion)
- Clear (unambiguous)
- Measurable (consistent scoring)
- Positively framed (assess qualities, not flaws)

### Examples

```python
unit_tests = [
    "Does the response cite specific arXiv paper IDs?",
    "Does the response distinguish between research findings and community opinions?",
    "Are potential limitations or caveats acknowledged?",
    "Is the response grounded in the retrieved sources?",
    "Does the response provide actionable next steps?",
]
```

### Scoring Rubric

Create detailed rubrics for nuanced scoring:

```python
unit_test = """
Does the response provide specific metrics?

Scoring Scale:
1: No metrics provided
2: Limited metrics without context
3: Basic metrics with some analysis
4: Clear metrics with detailed analysis
5: Comprehensive metrics with contextual analysis
"""
```

### API Usage

```python
from contextual import ContextualAI

client = ContextualAI(api_key="...")

result = client.lmunit.create(
    query="What is speculative decoding?",
    response="Speculative decoding is...",
    unit_test="Does the response explain the core mechanism?"
)

print(result.score)  # 1-5 scale
```

### Batch Evaluation

```python
import pandas as pd

def evaluate_responses(df, unit_tests):
    results = []
    for _, row in df.iterrows():
        for test in unit_tests:
            result = client.lmunit.create(
                query=row['query'],
                response=row['response'],
                unit_test=test
            )
            results.append({
                'query': row['query'],
                'test': test,
                'score': result.score
            })
    return pd.DataFrame(results)
```

---

## 13. Best Practices & Patterns

### Pattern: Multi-Source Search with Source-Aware Reranking

```yaml
nodes:
  search:
    type: SearchUnstructuredDataStep
    config:
      datastore_ids:
        - "arxiv-datastore"
        - "reddit-datastore"
      top_k: 50
      enable_query_decomposition: true
    input_mapping:
      query: __inputs__#query

  rerank:
    type: RerankRetrievalsStep
    config:
      rerank_top_k: 15
      rerank_instructions: |
        For technical questions: prioritize arxiv papers
        For practical questions: prioritize reddit posts
        For comparisons: balance both sources
    input_mapping:
      query: __inputs__#query
      retrievals: search#retrievals
```

### Pattern: Fallback Chain

```yaml
nodes:
  primary_search:
    type: SearchUnstructuredDataStep
    config:
      datastore_ids: ["primary-datastore"]
      top_k: 20
    input_mapping:
      query: __inputs__#query

  check_results:
    type: LanguageModelStep
    config:
      prompt: "Return 'sufficient' if results answer the query, else 'insufficient'"
    input_mapping:
      results: primary_search#retrievals

  route:
    type: ConditionalStep
    config:
      variable: sufficiency
      branches:
        - condition: "== sufficient"
          executable:
            type: ResponseGenerationStep
        - condition: "== insufficient"
          executable:
            type: subgraph:fallback_search
```

### Pattern: Query Classification + Routing

```yaml
nodes:
  classify:
    type: LanguageModelStep
    config:
      prompt: |
        Classify this query into one category:
        - THEORETICAL: algorithms, benchmarks, proofs, research
        - PRACTICAL: implementation, deployment, tooling, troubleshooting
        - COMPARISON: X vs Y, trade-offs, alternatives

        Query: {query}
        Category:
    input_mapping:
      query: __inputs__#query

  route:
    type: ConditionalStep
    config:
      variable: category
      branches:
        - condition: "== THEORETICAL"
          executable:
            type: subgraph:academic_search
        - condition: "== PRACTICAL"
          executable:
            type: subgraph:community_search
        - condition: "== COMPARISON"
          executable:
            type: subgraph:balanced_search
```

### Pattern: Iterative Refinement

```yaml
nodes:
  initial_search:
    type: SearchUnstructuredDataStep
    config:
      top_k: 30
    input_mapping:
      query: __inputs__#query

  assess:
    type: LanguageModelStep
    config:
      prompt: "Based on these results, what additional search would help? Return a refined query or 'DONE'"
    input_mapping:
      results: initial_search#retrievals
      query: __inputs__#query

  refined_search:
    type: SearchUnstructuredDataStep
    config:
      top_k: 20
    input_mapping:
      query: assess#refined_query

  merge:
    type: MergeStep
    input_mapping:
      initial: initial_search#retrievals
      refined: refined_search#retrievals
```

### Anti-Patterns to Avoid

| Anti-Pattern | Why It's Bad | Better Approach |
|--------------|--------------|-----------------|
| Hardcoding everything | No flexibility | Use config_overrides |
| Giant monolithic graph | Hard to debug | Break into subgraphs |
| No reranking | Poor result quality | Always rerank |
| Ignoring groundedness | Hallucination risk | Enable groundedness checks |
| Generic system prompts | Inconsistent outputs | Write specific guidelines |
| No evaluation | Unknown quality | Use LMUnit tests |

### Performance Tips

1. **Retrieve more, rerank fewer** — `top_k: 50` → `rerank_top_k: 15`
2. **Enable query decomposition** for complex questions
3. **Use parallel tool calls** in agentic workflows
4. **Cache subgraphs** for repeated patterns
5. **Stream UI** for better UX on long operations

---

## Quick Reference Card

### Essential YAML Skeleton

```yaml
inputs:
  query: str

outputs:
  response: str

nodes:
  search:
    type: SearchUnstructuredDataStep
    config:
      top_k: 50
      semantic_alpha: 0.8
      enable_query_decomposition: true
    input_mapping:
      query: __inputs__#query

  rerank:
    type: RerankRetrievalsStep
    config:
      rerank_top_k: 15
      rerank_instructions: "Your preferences here"
    input_mapping:
      query: __inputs__#query
      retrievals: search#retrievals

  generate:
    type: ResponseGenerationStep
    config:
      temperature: 0.3
      enable_groundedness_check: true
    input_mapping:
      query: __inputs__#query
      retrievals: rerank#retrievals

  __outputs__:
    type: output
    ui_output: true
    input_mapping:
      response: generate#response
```

### Wiring Cheat Sheet

| I want to... | Syntax |
|--------------|--------|
| Access graph input | `__inputs__#query` |
| Access node output | `node_name#output_key` |
| Use subgraph | `type: subgraph:name` |
| Dynamic config | `config_overrides: { key: node#output }` |
| Stream to UI | `ui_stream_types: { generation: true }` |

---

## Sources

- [Agent Composer Overview](https://docs.contextual.ai/reference/ac-overview.md)
- [YAML Format](https://docs.contextual.ai/reference/ac-yaml-format.md)
- [YAML Cheat Sheet](https://docs.contextual.ai/how-to-guides/ac-yaml-cheatsheet.md)
- [Step Reference](https://docs.contextual.ai/how-to-guides/ac-yaml-reference.md)
- [Node Type Reference](https://docs.contextual.ai/how-to-guides/ac-node-reference.md)
- [Agentic Workflows](https://docs.contextual.ai/reference/ac-agentic-workflows.md)
- [Conditional Nodes](https://docs.contextual.ai/reference/ac-conditional-nodes.md)
- [Subgraphs](https://docs.contextual.ai/reference/ac-subgraphs.md)
- [Input Mapping](https://docs.contextual.ai/reference/ac-input-mapping.md)
- [Config Overrides](https://docs.contextual.ai/reference/ac-config-overrides.md)
- [Tools Configuration](https://docs.contextual.ai/reference/ac-tools-config.md)
- [External Integrations](https://docs.contextual.ai/reference/ac-external-integrations.md)
- [MCP Integration](https://docs.contextual.ai/reference/ac-mcp-integration.md)
- [Webhooks](https://docs.contextual.ai/reference/ac-webhooks.md)
- [Code Execution](https://docs.contextual.ai/reference/ac-code-execution.md)
- [UI Stream Types](https://docs.contextual.ai/reference/ac-ui-stream-types.md)
- [LMUnit](https://docs.contextual.ai/how-to-guides/lmunit.md)
- [Chunking](https://docs.contextual.ai/reference/chunking.md)
- [System Prompts](https://docs.contextual.ai/reference/core-system-prompt.md)
- [Retrieval Parameters](https://docs.contextual.ai/reference/number-of-retrieved-chunks.md)
- [Search Weights](https://docs.contextual.ai/reference/semantic-search-weight.md)
