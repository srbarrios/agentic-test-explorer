# Guia Completa de Arquitectura: Agentic Test Explorer

> **Autor:** Generado como material didactico de Arquitectura del Software
>
> **Audiencia:** Ingenieros de software sin experiencia previa en LangChain/LangGraph
>
> **Proyecto:** [Agentic Test Explorer](https://github.com/srbarrios/agentic-test-explorer) -- un framework de QA exploratorio autonomo basado en agentes de IA

---

## Tabla de Contenidos

1. [Vision General del Proyecto](#1-vision-general-del-proyecto)
2. [Fundamentos Teoricos: LangChain y LangGraph](#2-fundamentos-teoricos-langchain-y-langgraph)
3. [El Grafo de Estados (StateGraph)](#3-el-grafo-de-estados-stategraph)
4. [Nodos, Aristas y Enrutamiento Condicional](#4-nodos-aristas-y-enrutamiento-condicional)
5. [El Patron Supervisor-Worker (Swarm)](#5-el-patron-supervisor-worker-swarm)
6. [El Estado Compartido: AgentState](#6-el-estado-compartido-agentstate)
7. [Herramientas (Tools) y el Decorador @tool](#7-herramientas-tools-y-el-decorador-tool)
8. [El Motor de Navegador: Record-and-Translate](#8-el-motor-de-navegador-record-and-translate)
9. [Los Modelos de Lenguaje (LLM): Proveedores Multiples](#9-los-modelos-de-lenguaje-llm-proveedores-multiples)
10. [Persistencia: Checkpoints y Store](#10-persistencia-checkpoints-y-store)
11. [El Sistema de Memoria de 4 Niveles](#11-el-sistema-de-memoria-de-4-niveles)
12. [Integraciones Externas: MCP y Skills](#12-integraciones-externas-mcp-y-skills)
13. [Las Misiones: Formato YAML y Enrutamiento](#13-las-misiones-formato-yaml-y-enrutamiento)
14. [Generacion de Tests desde Pull Requests](#14-generacion-de-tests-desde-pull-requests)
15. [Flujo de Ejecucion Completo](#15-flujo-de-ejecucion-completo)
16. [Patrones Arquitectonicos Clave](#16-patrones-arquitectonicos-clave)
17. [Glosario](#17-glosario)
18. [Referencias y Documentacion Oficial](#18-referencias-y-documentacion-oficial)

---

## 1. Vision General del Proyecto

### Que es Agentic Test Explorer?

Es un framework de **QA exploratorio autonomo** que utiliza agentes de inteligencia artificial para testear aplicaciones web de forma automatica. En lugar de escribir tests manuales, defines "misiones" en lenguaje natural y los agentes de IA navegan la aplicacion buscando bugs, como lo haria un tester humano con diferentes personalidades.

### Analogia para entenderlo

Imagina que contratas un equipo de testers QA:

- **Un usuario novato** que nunca ha visto la aplicacion
- **Un usuario experto** que conoce todos los atajos
- **Un "hacker" travieso** que intenta romper todo

Cada uno tiene su propia personalidad y estrategia de testing. Un **supervisor** les asigna tareas y decide quien testea que. Este proyecto replica exactamente ese equipo, pero con agentes de IA.

### Arquitectura de Alto Nivel

```
                    +------------------+
                    |   Misiones YAML  |
                    |  (lenguaje nat.) |
                    +--------+---------+
                             |
                    +--------v---------+
                    |      main.py     |
                    |  (CLI / Dispatch)|
                    +--------+---------+
                             |
              +--------------+--------------+
              |                             |
    +---------v---------+       +-----------v-----------+
    |   Standard Graph  |       |    Advanced Graph     |
    |   (3 personas)    |       |   (5 personas + exp.) |
    +---------+---------+       +-----------+-----------+
              |                             |
    +---------v-----------------------------v---------+
    |               SUPERVISOR NODE                    |
    |  (LLM decide que agente actua siguiente)         |
    +--+-------+--------+--------+--------+--------+--+
       |       |        |        |        |        |
       v       v        v        v        v        v
    [Agent1] [Agent2] [Agent3] [Agent4] [Agent5] [Explorer]
       |       |        |        |        |        |
       +-------+--------+--------+--------+--------+
                         |
              +----------v-----------+
              |   Browser Engine     |
              |  (Playwright)        |
              |  JSON intent -> exec |
              +----------+-----------+
                         |
              +----------v-----------+
              |   Action Tape        |
              |  (JSONL inmutable)   |
              +----------+-----------+
                         |
              +----------v-----------+
              |  .spec.ts generado   |
              |  (reproducible)      |
              +----------------------+
```

### Tecnologias Principales

| Tecnologia | Rol en el Proyecto | Documentacion |
|---|---|---|
| **LangGraph** | Orquestacion del grafo de agentes | [docs.langchain.com/oss/python/langgraph](https://docs.langchain.com/oss/python/langgraph/graph-api) |
| **LangChain** | Abstraccion de LLMs y herramientas | [docs.langchain.com/oss/python/langchain](https://docs.langchain.com/oss/python/langchain/tools) |
| **Playwright** | Automatizacion del navegador | [playwright.dev/python](https://playwright.dev/python/) |
| **Claude / Gemini** | Modelos de lenguaje (cerebro) | [docs.anthropic.com](https://docs.anthropic.com) / [ai.google.dev](https://ai.google.dev) |
| **MCP** | Protocolo de contexto para herramientas externas | [github.com/langchain-ai/langchain-mcp-adapters](https://github.com/langchain-ai/langchain-mcp-adapters) |

---

## 2. Fundamentos Teoricos: LangChain y LangGraph

### Que es LangChain?

**LangChain** es un framework de Python que facilita la construccion de aplicaciones sobre modelos de lenguaje (LLMs). Proporciona abstracciones para:

- **Chat Models:** Interfaces unificadas para distintos proveedores de LLM (Claude, GPT, Gemini, etc.)
- **Tools (Herramientas):** Funciones que el LLM puede "decidir llamar" durante su razonamiento
- **Chains:** Secuencias de pasos (prompt -> LLM -> parser -> accion)
- **Agents:** Sistemas donde el LLM elige dinamicamente que herramientas usar

> **Documentacion oficial:** [docs.langchain.com/oss/python/langchain](https://docs.langchain.com/oss/python/langchain/tools)

### Que es LangGraph?

**LangGraph** es una libreria construida *sobre* LangChain que permite definir flujos de trabajo como **grafos de estados**. Mientras que LangChain proporciona las piezas (LLMs, tools, prompts), LangGraph las conecta en flujos complejos con ciclos, bifurcaciones y persistencia.

**Analogia:** Si LangChain son los ladrillos, LangGraph es el plano del edificio.

**Conceptos clave de LangGraph:**

| Concepto | Descripcion | Analogia |
|---|---|---|
| **StateGraph** | El contenedor del flujo de trabajo | El tablero de un juego de mesa |
| **State (Estado)** | Los datos compartidos entre nodos | Las fichas y marcadores del juego |
| **Node (Nodo)** | Una funcion que transforma el estado | Un turno de juego |
| **Edge (Arista)** | Una conexion entre nodos | Las reglas de avance |
| **Conditional Edge** | Una conexion que depende del estado | "Si sacas 6, ve a..." |
| **Checkpointer** | Persistencia del estado entre ejecuciones | Guardar la partida |
| **Store** | Memoria a largo plazo entre hilos | La memoria del jugador |

> **Documentacion oficial:** [docs.langchain.com/oss/python/langgraph/graph-api](https://docs.langchain.com/oss/python/langgraph/graph-api)

### La Relacion entre Ambos

```
+----------------------------------------------------------+
|                      LangGraph                            |
|  +----------------------------------------------------+  |
|  |  StateGraph + Nodes + Edges + Checkpointer + Store |  |
|  |                                                    |  |
|  |  Usa internamente:                                 |  |
|  |  +----------------------------------------------+  |  |
|  |  |               LangChain                      |  |  |
|  |  |  ChatAnthropic | ChatGoogleGenerativeAI      |  |  |
|  |  |  @tool decorator | StructuredTool             |  |  |
|  |  |  BaseMessage | HumanMessage | AIMessage       |  |  |
|  |  +----------------------------------------------+  |  |
|  +----------------------------------------------------+  |
+----------------------------------------------------------+
```

---

## 3. El Grafo de Estados (StateGraph)

### Teoria: Que es un StateGraph?

Un `StateGraph` es la pieza central de LangGraph. Es un **grafo dirigido** donde:

- Cada **nodo** es una funcion asincrona que recibe el estado actual y devuelve actualizaciones
- Cada **arista** conecta nodos y define el orden de ejecucion
- El **estado** es un diccionario tipado (`TypedDict`) compartido entre todos los nodos

El modelo de ejecucion esta inspirado en **Pregel** (el sistema de Google para procesamiento de grafos a gran escala): los nodos se ejecutan en "super-pasos" y el estado se actualiza entre cada paso.

### Como se crea un StateGraph

```python
from langgraph.graph import StateGraph, END

# 1. Definir el esquema de estado
class MiEstado(TypedDict):
    mensajes: list
    siguiente_nodo: str

# 2. Crear el grafo
workflow = StateGraph(MiEstado)

# 3. Agregar nodos
workflow.add_node("nodo_a", funcion_a)
workflow.add_node("nodo_b", funcion_b)

# 4. Agregar aristas
workflow.add_edge("nodo_a", "nodo_b")
workflow.add_edge("nodo_b", END)

# 5. Definir punto de entrada
workflow.set_entry_point("nodo_a")

# 6. Compilar
app = workflow.compile()
```

### En este proyecto: `compile_swarm()` (graph_base.py:442)

Este proyecto no usa un grafo simple, sino un **grafo ciclico** donde el Supervisor puede redirigir la ejecucion a diferentes agentes multiples veces. Veamos como se construye:

```python
# Archivo: src/agentic_explorer/orchestration/graph_base.py

def compile_swarm(workflow, agent_registry: dict, checkpointer, store=None):
    """Wire agents -> Summarizer -> Supervisor -> conditional routing -> END"""
    from langgraph.graph import END

    agent_names = tuple(agent_registry.keys())

    # 1. Agregar el nodo Summarizer (compresion de mensajes)
    workflow.add_node("Summarizer", make_summarizer_node())

    # 2. Cada agente -> Summarizer (despues de actuar, se resumen mensajes)
    for agent_name in agent_names:
        workflow.add_edge(agent_name, "Summarizer")

    # 3. Summarizer -> Supervisor (el supervisor decide el siguiente paso)
    workflow.add_edge("Summarizer", "Supervisor")

    # 4. Supervisor -> agente O fin (enrutamiento condicional)
    route_map = {name: name for name in agent_names}
    route_map["FINISH"] = END

    workflow.add_conditional_edges(
        "Supervisor",                          # nodo origen
        lambda state: state["next_agent"],     # funcion de decision
        route_map,                             # mapa de destinos
    )

    # 5. El Supervisor es el punto de entrada
    workflow.set_entry_point("Supervisor")

    # 6. Compilar con persistencia
    return workflow.compile(checkpointer=checkpointer, store=store)
```

### Diagrama del grafo resultante

```
                    ENTRY
                      |
                      v
               +------+------+
               |  SUPERVISOR  |  <--- Decide que agente actua
               +------+------+
                      |
         +-----conditional_edges------+
         |            |               |
         v            v               v
   +-----------+ +-----------+  +-----------+
   | Agent #1  | | Agent #2  |  | Agent #N  |     "FINISH" -> END
   +-----------+ +-----------+  +-----------+
         |            |               |
         +------------+---------------+
                      |
                      v
              +--------------+
              | SUMMARIZER   |  <--- Comprime mensajes viejos
              +--------------+
                      |
                      v
               (vuelve a SUPERVISOR)
```

Este es un **grafo ciclico**: el Supervisor redirige a un agente, este actua, se resumen los mensajes, y el Supervisor vuelve a decidir. El ciclo continua hasta que el Supervisor decide `"FINISH"`.

> **Documentacion oficial sobre StateGraph:** [reference.langchain.com/python/langgraph/graph/state/StateGraph](https://reference.langchain.com/python/langgraph/graph/state/StateGraph)

---

## 4. Nodos, Aristas y Enrutamiento Condicional

### Teoria: Nodos

Un **nodo** en LangGraph es simplemente una funcion (sincrona o asincrona) que:

1. **Recibe** el estado actual como parametro
2. **Ejecuta** logica (llamar a un LLM, consultar una BD, ejecutar una herramienta...)
3. **Retorna** un diccionario con las claves del estado que quiere actualizar

```python
# Ejemplo basico de un nodo
async def mi_nodo(state: AgentState) -> dict:
    # Leer del estado
    mensajes = state["messages"]

    # Hacer algo
    resultado = await procesar(mensajes)

    # Retornar SOLO las claves que cambian
    return {"messages": [resultado]}
```

**Regla importante:** Un nodo NO necesita retornar todas las claves del estado. Solo retorna las que modifica. LangGraph se encarga de mergear las actualizaciones con el estado existente.

### Teoria: Aristas

LangGraph soporta dos tipos de aristas:

#### 1. Aristas Estaticas (`add_edge`)

Siempre van de A a B, sin condiciones:

```python
workflow.add_edge("new_user_agent", "Summarizer")
# Despues de que new_user_agent termine, SIEMPRE va a Summarizer
```

En este proyecto, todos los agentes tienen una arista estatica hacia el Summarizer:

```python
# graph_base.py:450-451
for agent_name in agent_names:
    workflow.add_edge(agent_name, "Summarizer")
```

#### 2. Aristas Condicionales (`add_conditional_edges`)

La decision de a donde ir depende del estado:

```python
workflow.add_conditional_edges(
    "Supervisor",                          # Nodo origen
    lambda state: state["next_agent"],     # Funcion que lee el estado
    {                                      # Mapa de destinos
        "new_user_agent": "new_user_agent",
        "power_user_agent": "power_user_agent",
        "adversarial_user_agent": "adversarial_user_agent",
        "FINISH": END,                     # END es una constante especial de LangGraph
    },
)
```

**Como funciona:** Despues de que el nodo `Supervisor` termina, LangGraph llama a la funcion lambda. Si `state["next_agent"]` es `"power_user_agent"`, la ejecucion va al nodo `power_user_agent`. Si es `"FINISH"`, el grafo termina.

### En este proyecto: Los tres tipos de nodos

El proyecto define tres fabricas de nodos en `graph_base.py`:

| Fabrica | Archivo:Linea | Proposito |
|---|---|---|
| `make_agent_node()` | graph_base.py:245 | Envuelve un agente compilado como nodo LangGraph |
| `make_supervisor_node()` | graph_base.py:291 | Crea el nodo supervisor que enruta al siguiente agente |
| `make_summarizer_node()` | graph_base.py:386 | Comprime mensajes para evitar desbordamiento de contexto |

> **Documentacion oficial sobre aristas condicionales:** [reference.langchain.com/python/langgraph/graph/state/StateGraph/add_conditional_edges](https://reference.langchain.com/python/langgraph/graph/state/StateGraph/add_conditional_edges)

---

## 5. El Patron Supervisor-Worker (Swarm)

### Teoria: Patrones Multi-Agente en LangGraph

LangGraph ofrece dos patrones principales para sistemas multi-agente:

#### Patron 1: Supervisor

Un LLM central (el "supervisor") recibe las tareas y delega a agentes especializados. Es como un jefe de equipo que decide quien trabaja en que.

```
     Usuario
        |
        v
   [Supervisor]  <---- LLM que decide
    /    |    \
   v     v     v
[Ag1] [Ag2] [Ag3]
```

> **Documentacion oficial:** [github.com/langchain-ai/langgraph-supervisor-py](https://github.com/langchain-ai/langgraph-supervisor-py)

#### Patron 2: Swarm

No hay supervisor central. Los agentes se pasan el control directamente entre ellos usando `Command(goto=...)`. Es como un equipo auto-organizado.

> **Documentacion oficial:** [github.com/langchain-ai/langgraph-swarm-py](https://github.com/langchain-ai/langgraph-swarm-py)

#### Este proyecto: Supervisor Custom

Este proyecto implementa una **variante personalizada del patron Supervisor**. No usa la libreria `langgraph-supervisor` directamente, sino que construye su propio supervisor con logica adicional:

- **Control de pasos:** Resetea cuando se excede un limite
- **Contexto de memoria:** Inyecta conocimiento de sesiones pasadas
- **Contexto de exploracion:** Informa que areas ya se visitaron

### El Nodo Supervisor: `make_supervisor_node()` (graph_base.py:291)

Veamos como funciona el supervisor paso a paso:

```python
def make_supervisor_node(llm, agent_names, app_url, max_steps, ...):

    # 1. Definir el esquema de respuesta estructurada
    routing_schema = {
        "title": "SupervisorRouting",
        "type": "object",
        "properties": {
            "next": {
                "type": "string",
                "enum": [*agent_names, "FINISH"]  # Solo puede elegir estos valores
            },
        },
        "required": ["next"],
    }

    # 2. Crear un LLM que SOLO responde con JSON valido
    routing_llm = llm.with_structured_output(
        schema=routing_schema,
        method="function_calling"
    )

    async def supervisor_node(state: AgentState, *, store=None) -> dict:
        # 3. Incrementar contador de pasos
        current_step = state.get("step_count", 0) + 1

        # 4. Si se excede el limite, resetear y forzar exploracion nueva
        if current_step > max_steps:
            reset_msg = HumanMessage(content=(
                f"[STEP LIMIT] Navigate back to {app_url} and pick "
                "a COMPLETELY DIFFERENT area."
            ))
            current_step = 1  # Reset

        # 5. Consultar memoria a largo plazo (paginas conocidas, bugs previos)
        if store:
            memory_context = await format_memory_context(store, url_hash)

        # 6. Construir el "brief" de enrutamiento (contexto compacto)
        routing_context = _build_routing_context(state, extra_messages, memory_context)

        # 7. Pedirle al LLM que decida
        decision = await routing_llm.ainvoke([
            SystemMessage(content=supervisor_prompt),
            HumanMessage(content=f"...{routing_context}...\nWhich agent should act next?")
        ])

        # 8. Retornar la decision
        return {"next_agent": decision["next"], "step_count": current_step}

    return supervisor_node
```

**Punto clave:** El metodo `with_structured_output()` fuerza al LLM a responder **solo** con un JSON que contenga la clave `"next"` con uno de los valores permitidos. Esto elimina la ambiguedad: el LLM no puede inventar agentes que no existen.

### Los dos tipos de grafos

El proyecto define dos grafos segun la complejidad de la mision:

#### Grafo Estandar (standard_graph.py): 3 Personas

| Agente | Rol | Enfoque |
|---|---|---|
| `new_user_agent` | Usuario Novato | Onboarding, descubrimiento, estados por defecto |
| `power_user_agent` | Usuario Experto | Atajos, operaciones masivas, filtros avanzados |
| `adversarial_user_agent` | Chaos Monkey | Inputs invalidos, inyeccion SQL, clics rapidos |

#### Grafo Avanzado (advanced_graph.py): 5 Personas + Explorador

| Agente | Rol | Enfoque |
|---|---|---|
| `accessibility_user_agent` | Accesibilidad | WCAG, navegacion por teclado, lectores de pantalla |
| `data_heavy_user_agent` | Datos Pesados | Archivos grandes, muchos registros, performance |
| `impatient_user_agent` | Usuario Impaciente | Cancelar a mitad, refrescar durante envios, race conditions |
| `returning_user_agent` | Usuario que Vuelve | Sesiones caducadas, paginas cacheadas, tokens expirados |
| `explorer_agent` | Explorador Autonomo | Chaos total: filtros, dropdowns, toggles, valores limite |

### Como se construye un agente

Cada agente se crea con `create_agent()` de LangChain, que recibe un LLM, herramientas y un prompt de sistema:

```python
# standard_graph.py:90-92
agent_registry[agent_name] = create_agent(
    llm,
    tools=dom_tools,                          # Herramientas disponibles
    system_prompt=SystemMessage(content=       # Personalidad del agente
        make_browser_agent_prompt(role, app_context, focus)
    )
)
```

El prompt de sistema combina:
- **Rol:** "You are the New User / First-Timer Persona"
- **Contexto de app:** "The application under test is 'Uyuni', accessible at ..."
- **Enfoque:** "Test onboarding, discoverability, default states..."
- **Reglas del navegador:** BROWSER_AGENT_RULES (politica de selectores, acciones permitidas, etc.)

---

## 6. El Estado Compartido: AgentState

### Teoria: TypedDict y Reducers

En LangGraph, el estado compartido se define como un `TypedDict` de Python. Cada campo puede tener un **reducer** (funcion reductora) que determina como se combinan las actualizaciones.

**Sin reducer:** La actualizacion simplemente **sobreescribe** el valor anterior.

**Con reducer:** La funcion reducer recibe `(valor_antiguo, valor_nuevo)` y devuelve el valor combinado.

```python
from typing import Annotated
import operator

class MiEstado(TypedDict):
    # Con reducer operator.add: las listas se CONCATENAN
    mensajes: Annotated[list, operator.add]

    # Sin reducer: el valor se SOBREESCRIBE
    nombre: str
```

### En este proyecto: AgentState (graph_base.py:55)

```python
class AgentState(TypedDict):
    # Mensajes del chat. Reducer: operator.add -> se acumulan
    messages: Annotated[Sequence[BaseMessage], operator.add]

    # Nombre del siguiente agente. Sin reducer -> se sobreescribe
    next_agent: str

    # Tape de acciones del navegador. Reducer custom -> maximo 50 entradas
    action_tape: Annotated[List[Dict], _bounded_tape_reducer]

    # Contador de pasos. Reducer lambda -> siempre el ultimo valor
    step_count: Annotated[int, lambda _old, new: new]

    # Bugs encontrados. Reducer: operator.add -> se acumulan
    bugs_found: Annotated[List[str], operator.add]

    # URLs visitadas. Reducer: operator.add -> se acumulan
    explored_paths: Annotated[List[str], operator.add]
```

### Explicacion campo por campo

#### `messages` -- Historial de conversacion

```python
messages: Annotated[Sequence[BaseMessage], operator.add]
```

- **Tipo:** Lista de `BaseMessage` (clase base de LangChain para mensajes)
- **Reducer:** `operator.add` -- cada nodo puede agregar mensajes y estos se concatenan al historial existente
- **Subtipos de mensaje usados:**
  - `HumanMessage`: La mision del usuario o directivas del supervisor
  - `AIMessage`: Respuestas/razonamiento del LLM
  - `ToolMessage`: Resultados de herramientas ejecutadas
  - `SystemMessage`: Instrucciones del sistema o resumenes comprimidos
  - `RemoveMessage`: Mensaje especial para eliminar mensajes del historial (usado por el Summarizer)

#### `next_agent` -- Enrutamiento

```python
next_agent: str
```

- **Sin reducer** -- el Supervisor lo sobreescribe en cada ciclo
- Contiene el nombre del proximo agente (`"new_user_agent"`, `"FINISH"`, etc.)
- Es leido por la arista condicional para decidir a donde ir

#### `action_tape` -- Registro de acciones del navegador

```python
action_tape: Annotated[List[Dict], _bounded_tape_reducer]
```

- **Reducer custom** (`_bounded_tape_reducer`):

```python
def _bounded_tape_reducer(old, new):
    combined = (old or []) + (new or [])
    return combined[-50:]  # Solo las 50 mas recientes
```

- **Por que limitar a 50?** El Action Tape completo se persiste en disco como JSONL. En el estado solo mantenemos las mas recientes para que el contexto del LLM no se desborde. Esto es un **patron de ventana deslizante**.

#### `step_count` -- Prevencion de bucles infinitos

```python
step_count: Annotated[int, lambda _old, new: new]
```

- **Reducer lambda:** Siempre retorna el nuevo valor (sobreescribe)
- El Supervisor lo incrementa en cada ciclo
- Cuando excede `max_steps`, se resetea el agente al inicio de la aplicacion

#### `bugs_found` y `explored_paths` -- Tracking

```python
bugs_found: Annotated[List[str], operator.add]
explored_paths: Annotated[List[str], operator.add]
```

- Ambos usan `operator.add` para acumular datos de todas las iteraciones
- Alimentan el contexto del Supervisor para evitar redundancia y guiar la exploracion

> **Documentacion oficial sobre estado y reducers:** [docs.langchain.com/oss/python/langgraph/graph-api](https://docs.langchain.com/oss/python/langgraph/graph-api)

---

## 7. Herramientas (Tools) y el Decorador @tool

### Teoria: Que es una Tool en LangChain?

Una **Tool** (herramienta) es una funcion que un LLM puede "decidir llamar" durante su razonamiento. El flujo es:

1. Le describes la herramienta al LLM (nombre + descripcion + parametros)
2. El LLM razona y decide: "necesito llamar a la herramienta X con estos argumentos"
3. El framework ejecuta la herramienta y devuelve el resultado al LLM
4. El LLM incorpora el resultado en su razonamiento

Este ciclo se conoce como **ReAct** (Reasoning + Acting).

### Formas de definir herramientas

#### 1. Decorador `@tool` (la mas comun)

```python
from langchain_core.tools import tool

@tool
async def mi_herramienta(parametro: str) -> str:
    """Descripcion que el LLM leera para decidir si usar esta herramienta.

    Args:
        parametro: Explicacion del parametro.
    """
    resultado = await hacer_algo(parametro)
    return resultado
```

**Puntos clave:**
- El **docstring** es crucial: el LLM lo lee para decidir cuando usar la herramienta
- Los **type hints** definen el esquema de entrada automaticamente
- El **nombre** de la funcion es el nombre de la herramienta

#### 2. `StructuredTool.from_function()` (mas control)

```python
from langchain_core.tools import StructuredTool

mi_tool = StructuredTool.from_function(
    func=mi_funcion,
    name="nombre_custom",
    description="Descripcion custom",
    args_schema=MiPydanticModel,
)
```

> **Documentacion oficial sobre herramientas:** [docs.langchain.com/oss/python/langchain/tools](https://docs.langchain.com/oss/python/langchain/tools)

### Herramientas en este proyecto

El proyecto define varias herramientas especializadas, todas como fabricas (factory functions) que capturan recursos del entorno (como la `page` de Playwright):

#### Herramientas del navegador (engine.py)

| Herramienta | Archivo:Linea | Descripcion |
|---|---|---|
| `execute_browser_command` | engine.py:419 | Ejecuta UNA accion JSON en el navegador |
| `get_dom_snapshot` | engine.py:498 | Lee el DOM actual (solo lectura) |
| `generate_reproduction_spec` | engine.py:609 | Genera un .spec.ts desde el Action Tape |

#### Herramientas de captura (custom_tools.py)

| Herramienta | Descripcion |
|---|---|
| `capture_bug_screenshot` | Captura screenshot cuando se detecta un bug |
| `analyze_visual_state` | Envia screenshot a LLM con vision para validacion |

#### Herramientas de memoria (memory.py + Langmem)

| Herramienta | Origen | Descripcion |
|---|---|---|
| `recall_past_findings` | memory.py (custom) | Busqueda semantica (o keyword) de bugs, sesiones y quirks |
| `record_observation` | Langmem `create_manage_memory_tool` | Registra observaciones proactivas para futuras sesiones |

### Ejemplo detallado: `execute_browser_command`

Esta es la herramienta mas importante del proyecto. Veamos como esta construida:

```python
# engine.py:414
def get_browser_command_tool(page: Page):
    """Factory: captura la referencia a la pagina de Playwright."""

    @tool
    async def execute_browser_command(command_json: str, config: RunnableConfig) -> str:
        """Execute ONE browser action described as strict JSON...

        Argument `command_json` MUST be a JSON object string, for example:
            {"action": "navigate", "url": "https://example.com/app"}
            {"action": "click", "selector": "[data-test-subj='submitButton']"}
        """
        # 1. Obtener thread_id del config (para el Action Tape)
        thread_id = config.get("configurable", {}).get("thread_id", "default")

        # 2. Parsear el JSON del agente
        parsed = json.loads(command_json)
        action = parsed["action"]
        params = {k: v for k, v in parsed.items() if k != "action"}

        # 3. Validar selector (rechaza XPath y selectores fragiles)
        selector_error = _validate_selector(params.get("selector", ""))
        if selector_error:
            return f"STATUS: ERROR\nERROR: {selector_error}"

        # 4. Ejecutar la accion con Playwright
        resultado = await _dispatch(page, action, params)

        # 5. Capturar snapshot del DOM despues de la accion
        snap = await extract_dom_snapshot(page)

        # 6. Registrar en el Action Tape (inmutable)
        _append_tape(thread_id, result.to_tape_entry())

        # 7. Devolver resultado formateado al agente
        return f"STATUS: OK\nRESULT: {resultado}\nDOM_SNAPSHOT:\n{snapshot_formateado}"

    return execute_browser_command
```

**Patron Factory:** La funcion `get_browser_command_tool(page)` es una **fabrica** que retorna la herramienta con la `page` de Playwright "capturada" en el closure. Esto permite que la herramienta acceda a la pagina sin que sea un parametro del agente.

**RunnableConfig:** El parametro `config: RunnableConfig` es inyectado automaticamente por LangGraph. Contiene metadatos como el `thread_id` de la mision actual. Esto permite que la herramienta escriba en el Action Tape correcto.

---

## 8. El Motor de Navegador: Record-and-Translate

### Teoria: Separacion Cerebro / Manos

El proyecto implementa un patron arquitectonico llamado **Record-and-Translate** (Grabar y Traducir), que separa dos responsabilidades:

```
+--------------------+    JSON intent    +-------------------+
|    CEREBRO (IA)    | ----------------> |   MANOS (Engine)  |
| Agentes LangGraph  |                  |   Playwright       |
| Deciden QUE hacer   |                  |   Ejecuta HOW      |
+--------------------+                  +-------------------+
                                                |
                                         Registra todo en
                                                |
                                                v
                                        +-------------------+
                                        |   Action Tape     |
                                        |   (JSONL inmutable)|
                                        +-------------------+
                                                |
                                         Se traduce a
                                                v
                                        +-------------------+
                                        |   .spec.ts        |
                                        |   (Playwright)    |
                                        +-------------------+
```

**Por que esta separacion?**

1. **Determinismo:** Las acciones del navegador son deterministas y reproducibles
2. **Depurabilidad:** El Action Tape es un log inmutable de todo lo que se hizo
3. **Reproducibilidad:** El tape se traduce a un script Playwright ejecutable
4. **Seguridad:** Los agentes no tienen acceso directo a Playwright -- solo pueden emitir intenciones JSON validadas

### Acciones Soportadas

El motor solo acepta un conjunto fijo de acciones (engine.py:253):

```python
ALLOWED_ACTIONS = {
    "navigate",       # {"action": "navigate", "url": "https://..."}
    "click",          # {"action": "click", "selector": "[data-test-subj='btn']"}
    "fill",           # {"action": "fill", "selector": "input#name", "value": "Oscar"}
    "press",          # {"action": "press", "selector": "input", "key": "Enter"}
    "select_option",  # {"action": "select_option", "selector": "select", "value": "opt1"}
    "hover",          # {"action": "hover", "selector": ".menu-trigger"}
    "wait_for",       # {"action": "wait_for", "selector": ".content", "state": "visible"}
    "scroll",         # {"action": "scroll", "selector": ".panel"}  o {"y": 400}
    "extract_text",   # {"action": "extract_text", "selector": "h1"}
    "snapshot",       # {"action": "snapshot"}  -- solo lee el DOM
    "check_page_health",  # Detecta banners de error y spinners
}
```

### Politica de Selectores Resilientes

Uno de los patrones mas importantes del proyecto es la **validacion de selectores** (engine.py:305):

```python
# Selectores RECHAZADOS (fragiles, dependen de la posicion en el DOM):
_BRITTLE_SELECTOR_PATTERNS = re.compile(r"""
    (^/)                     # XPath: /html/body/div[1]
    | (/{2})                 # XPath descendiente: //div[@class='x']
    | (:nth-child\s*\()      # CSS posicional: li:nth-child(3)
    | (:nth-of-type\s*\()    # CSS posicional: div:nth-of-type(2)
""")

# Selectores PREFERIDOS (resilientes, sobreviven refactorizacion del DOM):
# 1. data-test-subj -> [data-test-subj='myButton']
# 2. aria-label     -> [aria-label='Search bar']
# 3. Texto visible  -> button:has-text('Save')
# 4. Rol semantico  -> role=button[name='Submit']
```

**Por que?** Los selectores como `div:nth-child(3)` o `//div[1]/span[2]` se rompen cuando alguien agrega un `<div>` antes. Los selectores basados en atributos semanticos (`data-test-subj`, `aria-label`) son estables.

### Extraccion del DOM (Snapshot)

Para que el agente "vea" la pagina, el motor extrae un snapshot compacto del DOM usando dos estrategias (engine.py:172):

#### Estrategia primaria: Arbol de Accesibilidad de Playwright

```python
ax_root = await page.accessibility.snapshot(interesting_only=True)
```

Playwright proporciona un **arbol de accesibilidad** que ya filtra scripts, estilos y elementos invisibles. Solo muestra lo que un usuario real (o un lector de pantalla) puede percibir.

Ejemplo de salida:
```
* [0] role=link name='Home'
* [1] role=button name='Search'
* [2] role=textbox name='Email' value='user@example.com'
- [3] role=heading name='Welcome back'
```

#### Estrategia fallback: DOM Walk con JavaScript

Si el arbol de accesibilidad falla (por ejemplo, por restricciones del navegador), se ejecuta un script JavaScript que recorre el DOM filtrando tags no utiles (`<script>`, `<style>`, `<svg>`, etc.).

### Action Tape: Registro Inmutable

Cada accion ejecutada se registra en dos lugares:

1. **En memoria:** `_ACTION_TAPES[thread_id]` (diccionario en RAM)
2. **En disco:** `report_{thread_id}/action_tape.jsonl` (un JSON por linea)

```python
def _append_tape(thread_id: str, entry: Dict[str, Any]) -> None:
    get_action_tape(thread_id).append(entry)
    with open(_tape_path(thread_id), "a") as f:
        f.write(json.dumps(entry) + "\n")
```

Cada entrada contiene:
```json
{
    "ts": 1716000000.123,
    "action": "click",
    "params": {"selector": "[data-test-subj='saveBtn']"},
    "ok": true,
    "duration_ms": 142,
    "result": "clicked [data-test-subj='saveBtn']",
    "error": null,
    "page_url": "https://app.example.com/settings",
    "page_title": "Settings"
}
```

### Generacion de .spec.ts

El Action Tape se traduce a un script Playwright ejecutable (engine.py:556):

```typescript
// Auto-generated reproduction for bug: Error banner on save
// Generated: 2026-05-17 14:30:00
import { test, expect } from '@playwright/test';

test.use({ storageState: 'auth.json' });

test('reproduce: Error banner on save', async ({ page }) => {
    test.setTimeout(120_000);
    await page.goto('https://app.example.com/settings');
    await page.fill('[aria-label="Name"]', 'Test User');
    await page.click('[data-test-subj="saveBtn"]');
    // FAILED AT RECORD TIME: check_page_health
    //   Error: HEALTH: ERROR - 1 error banner(s): Internal Server Error
});
```

Las acciones que fallaron se incluyen como **comentarios** para documentar el bug.

> **Documentacion oficial de Playwright:** [playwright.dev/python](https://playwright.dev/python/)

---

## 9. Los Modelos de Lenguaje (LLM): Proveedores Multiples

### Teoria: Chat Models en LangChain

LangChain abstrae diferentes proveedores de LLM bajo una interfaz comun. Los mas relevantes para este proyecto:

#### ChatAnthropic (Claude)

```python
from langchain_anthropic import ChatAnthropic

llm = ChatAnthropic(
    model="claude-sonnet-4-6",
    temperature=0,
    api_key="sk-..."
)
```

> **Documentacion oficial:** [docs.langchain.com/oss/python/integrations/chat/anthropic](https://docs.langchain.com/oss/python/integrations/chat/anthropic)

#### ChatGoogleGenerativeAI (Gemini)

```python
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
)
```

> **Documentacion oficial:** [reference.langchain.com/python/integrations/langchain_google_genai/ChatGoogleGenerativeAI](https://reference.langchain.com/python/integrations/langchain_google_genai/ChatGoogleGenerativeAI/)

### En este proyecto: Fabrica Multi-Proveedor (llm.py)

El modulo `llm.py` implementa una **fabrica** que detecta automaticamente que proveedor usar basandose en las credenciales disponibles:

```python
def _detect_provider() -> str:
    """Orden de prioridad para auto-deteccion:"""
    # 1. Variable de entorno explicita
    if os.getenv("LLM_PROVIDER") in ("gemini", "claude"):
        return env_provider

    # 2. API Key de Anthropic -> Claude directo
    if os.getenv("ANTHROPIC_API_KEY"):
        return "claude"

    # 3. Configuracion de Vertex AI en ~/.claude/settings.json
    if _claude_vertex_config() is not None:
        return "claude"

    # 4. API Key de Google -> Gemini
    if os.getenv("GOOGLE_API_KEY"):
        return "gemini"

    # 5. Credenciales OAuth de Gemini
    if Path("~/.gemini/oauth_creds.json").is_file():
        return "gemini"

    raise RuntimeError("No LLM credentials found.")
```

### Seleccion de Modelo

El proyecto selecciona modelos diferentes segun el metodo de autenticacion:

| Proveedor | Metodo Auth | Modelo por Defecto | Razon |
|---|---|---|---|
| Claude | API Key directa | `claude-haiku-4-5` | Economico para desarrollo |
| Claude | Vertex AI (GCP) | `claude-haiku-4-5` | Economico para desarrollo |
| Gemini | API Key | `gemini-2.5-flash` | Economico, rapido |
| Gemini | OAuth | `gemini-3.1-flash` | Rapido con suscripcion |

### Structured Output: Como el LLM responde con JSON

El Supervisor necesita que el LLM responda con un JSON exacto (`{"next": "agent_name"}`). LangChain lo logra con `with_structured_output()`:

```python
routing_llm = llm.with_structured_output(
    schema={
        "properties": {
            "next": {"type": "string", "enum": ["new_user_agent", "FINISH"]}
        }
    },
    method="function_calling"
)

# El LLM SOLO puede responder con:
# {"next": "new_user_agent"}  o  {"next": "FINISH"}
```

Internamente, esto usa **function calling** del LLM -- una capacidad nativa de modelos como Claude y Gemini donde el modelo genera llamadas a funciones con esquemas estrictos.

---

## 10. Persistencia: Checkpoints y Store

### Teoria: Dos tipos de memoria en LangGraph

LangGraph distingue dos tipos de persistencia:

#### 1. Checkpointer: Memoria a corto plazo (por hilo)

Guarda **snapshots automaticos** del estado del grafo en cada paso. Permite:

- **Reanudar:** Si la ejecucion falla, puede continuar desde el ultimo checkpoint
- **Time travel:** Volver a un paso anterior del grafo
- **Human-in-the-loop:** Pausar la ejecucion para aprobacion humana

```python
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

checkpointer = AsyncSqliteSaver.from_conn_string("checkpoints.db")
graph = workflow.compile(checkpointer=checkpointer)
```

Cada ejecucion se identifica por un `thread_id` unico. Dos ejecuciones con el mismo `thread_id` comparten el mismo historial de checkpoints.

> **Documentacion oficial:** [docs.langchain.com/oss/python/langgraph/persistence](https://docs.langchain.com/oss/python/langgraph/persistence)

#### 2. Store: Memoria a largo plazo (entre hilos)

Un **almacen key-value** organizado por namespaces. Persiste informacion que debe sobrevivir entre diferentes ejecuciones (diferentes `thread_id`s).

```python
from langgraph.store.memory import InMemoryStore

store = InMemoryStore()
graph = workflow.compile(checkpointer=checkpointer, store=store)
```

Dentro de un nodo, se accede al store asi:

```python
async def mi_nodo(state, *, store=None):
    # Escribir
    await store.aput(
        namespace=("app", "abc123", "pages"),
        key="home",
        value={"url": "/", "visit_count": 5}
    )

    # Leer
    item = await store.aget(("app", "abc123", "pages"), "home")
    print(item.value)  # {"url": "/", "visit_count": 5}

    # Buscar
    results = await store.asearch(("app", "abc123", "pages"), limit=10)
```

### En este proyecto

El proyecto usa **ambos** mecanismos:

```python
# main.py (simplificado)
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.store.memory import AsyncSqliteStore  # Store con SQLite

async with AsyncSqliteSaver.from_conn_string("checkpoints.db") as checkpointer:
    async with AsyncSqliteStore.from_conn_string("memory.db") as store:
        graph = await build_graph(
            ...,
            checkpointer=checkpointer,
            store=store,
        )
```

| Mecanismo | Base de datos | Que guarda | Ciclo de vida |
|---|---|---|---|
| Checkpointer | `checkpoints.db` | Estado del grafo en cada paso | Por mision (thread_id) |
| Store | `memory.db` | Conocimiento aprendido | Permanente (entre sesiones) |

---

## 11. El Sistema de Memoria de 4 Niveles

### Teoria: Memoria Cognitiva en Agentes

Este proyecto implementa un sistema de memoria inspirado en la psicologia cognitiva humana, con cuatro niveles. Las operaciones basadas en LLM (reflexion procedural, observaciones de agentes) utilizan el **SDK Langmem** (`langmem`).

```
+------------------------------------------------------------------+
|                    MEMORIA DEL SISTEMA                            |
|                 (LangGraph Store + Langmem SDK)                   |
|                                                                   |
|  +--------------------+  +--------------------+                   |
|  |  MEMORIA SEMANTICA |  |  MEMORIA EPISODICA |                  |
|  |  (hechos)          |  |  (experiencias)    |                   |
|  |                    |  |                    |                   |
|  |  - Paginas         |  |  - Sesiones        |                  |
|  |  - Selectores      |  |  - Catalogo bugs   |                  |
|  |  - Quirks          |  |                    |                   |
|  |  - Observaciones * |  |                    |                   |
|  +--------------------+  +--------------------+                   |
|                                                                   |
|  +--------------------+  +--------------------+                   |
|  |  MEMORIA PROCEDU.  |  |  PRIORIZACION      |                  |
|  |  (habilidades)     |  |  (que testear)     |                   |
|  |                    |  |                    |                   |
|  |  - Prompt optim. * |  |  - Paginas riesgo  |                  |
|  |  - Reglas routing  |  |  - Bugs recurrentes|                  |
|  |  - Que evitar      |  |                    |                   |
|  +--------------------+  +--------------------+                   |
+------------------------------------------------------------------+
  * = Powered by Langmem SDK
```

### Organizacion por Namespaces

Toda la memoria se organiza en el Store con namespaces jerarquicos (memory.py:1-14):

```
("app", "{url_hash}", "pages")               # Semantica: paginas conocidas
("app", "{url_hash}", "selectors")           # Semantica: fiabilidad de selectores
("app", "{url_hash}", "quirks")              # Semantica: comportamientos raros de la app
("app", "{url_hash}", "agent_observations")  # Semantica: observaciones de agentes (Langmem)
("episodes", "{url_hash}", "sessions")       # Episodica: resumen de sesiones
("episodes", "{url_hash}", "bugs")           # Episodica: catalogo de bugs
("procedures", "{url_hash}", "agent_prompts")   # Procedural: prompts optimizados (Langmem)
("procedures", "{url_hash}", "routing_rules")   # Procedural: reglas del supervisor
```

El `url_hash` es un hash SHA-256 de la URL de la app (`app_url_hash()`, memory.py:25), lo que permite que la memoria sea **por aplicacion**: si testeas dos apps diferentes, cada una tiene su propia memoria.

### Nivel 1: Memoria Semantica (Hechos)

La memoria semantica almacena **conocimiento factual** sobre la aplicacion bajo test.

#### Paginas conocidas (`update_page_knowledge`, memory.py:35)

Cada vez que un agente navega exitosamente a una pagina:

```python
await update_page_knowledge(store, url_hash, page_url="https://app.com/settings", page_title="Settings")
```

Se registra:
```json
{
    "url": "/settings",
    "visit_count": 3,
    "title": "Settings",
    "last_seen": "2026-05-17T10:30:00Z"
}
```

**Para que?** El Supervisor sabe que paginas ya se exploraron y puede dirigir agentes a areas no visitadas.

#### Fiabilidad de selectores (`track_selector`, memory.py:61)

Cada selector usado se rastrea con su tasa de exito/fallo:

```python
await track_selector(store, url_hash, selector="[data-test-subj='saveBtn']", success=True)
await track_selector(store, url_hash, selector=".old-class-name", success=False)
```

**Para que?** Los selectores con alta tasa de fallo indican areas fragiles de la UI.

#### Quirks de la aplicacion (`record_quirk`, memory.py:94)

Comportamientos inesperados descubiertos durante el testing:

```python
await record_quirk(
    store, url_hash,
    description="La pagina de settings tarda 8 segundos en cargar despues de guardar",
    page="/settings",
    category="performance",
    discovered_by="impatient_user_agent"
)
```

### Nivel 2: Memoria Episodica (Experiencias)

La memoria episodica almacena **que paso en cada sesion de testing**.

#### Resumenes de sesiones (`write_session_summary`, memory.py:203)

Al finalizar cada mision:

```python
await write_session_summary(
    store, url_hash,
    thread_id="test_login_01",
    mission_prompt="Test the login flow...",
    action_tape=[...],           # Todas las acciones ejecutadas
    bugs_found=["Error on submit"],
    explored_paths=["/login", "/dashboard"],
)
```

Se registra:
```json
{
    "thread_id": "test_login_01",
    "total_actions": 25,
    "successful_actions": 22,
    "bugs_found": 1,
    "pages_covered": ["/login", "/dashboard"],
    "outcome": "bugs_found",
    "completed_at": "2026-05-17T11:00:00Z"
}
```

#### Catalogo de bugs (`catalog_bug`, memory.py:239)

Los bugs se deduplicican usando un fingerprint determinista:

```python
def _bug_fingerprint(summary, page):
    normalized = f"{page}:{summary[:100]}".strip().lower()
    return hashlib.md5(normalized.encode()).hexdigest()[:12]
```

Si el mismo bug aparece en multiples sesiones, se incrementa el `seen_count` en lugar de duplicar la entrada.

### Nivel 3: Memoria Procedural (Habilidades)

La memoria procedural almacena **que funciono y que no** para mejorar las futuras sesiones.

#### Reflexion post-batch (`update_procedural_memory`, memory.py — powered by Langmem)

Despues de ejecutar todas las misiones de un batch, el sistema usa el **prompt optimizer de Langmem** (`create_prompt_optimizer(kind="prompt_memory")`) para reflexionar sobre los resultados. El optimizer recibe los resumenes de sesiones como trayectorias y genera prompts optimizados para cada agente y para las reglas de routing del supervisor.

```python
from langmem import create_prompt_optimizer

optimizer = create_prompt_optimizer(llm, kind="prompt_memory")

# Para cada agente, optimiza su prompt basandose en las trayectorias
optimized = await optimizer.ainvoke({
    "trajectories": [(trajectory_messages, {"context": "batch review"})],
    "prompt": current_agent_prompt,
})
```

El optimizer analiza los resultados y genera prompts mejorados que se inyectan en los agentes en la siguiente ejecucion:

```
LEARNED FROM PAST SESSIONS:
Key observations:
- The settings page uses a custom save mechanism with 5s debounce
- Error banners appear inside a shadow DOM container

Effective strategies:
- Testing with very long strings (>1000 chars) reliably triggers validation bugs
- Rapid form submission exposes race conditions

Avoid:
- Testing navigation menu items — they are fully static and bug-free
```

#### Suplementos de prompt por agente (`get_agent_prompt_supplement`, memory.py:373)

Cada agente recibe un suplemento personalizado basado en lo aprendido:

```python
supplement = await get_agent_prompt_supplement(store, url_hash, "adversarial_user_agent")
if supplement:
    focus = f"{focus}\n\n{supplement}"
```

Esto hace que los agentes sean **mas inteligentes con el tiempo**: no repiten errores de sesiones pasadas.

### Nivel 4: Priorizacion (Que testear primero)

El sistema calcula un **score de riesgo** para cada pagina (memory.py:643):

```python
async def prioritize_pages(store, url_hash) -> str:
    page_scores = {}

    # Paginas con mas bugs -> mayor riesgo
    for bug in bugs:
        page_scores[page] += bug.seen_count * 2

    # Paginas con quirks confirmados -> mayor riesgo
    for quirk in quirks:
        page_scores[page] += quirk.confirmed_count

    # Selectores con fallos -> area fragil
    for selector in selectors:
        page_scores[page] += selector.failure_count * 1.5
```

Resultado inyectado al Supervisor:
```
HIGH_RISK_PAGES (prioritize testing these areas):
- /settings (risk score: 12)
- /user/profile (risk score: 8)
- /reports/export (risk score: 5)
```

### Herramientas de Consulta y Observacion

#### `recall_past_findings` (busqueda semantica)

Los agentes pueden consultar la memoria a largo plazo durante su ejecucion. Cuando el store tiene un indice de embeddings configurado, la busqueda usa **similitud vectorial semantica**; si no, hace fallback a busqueda por palabras clave:

```python
@tool
async def recall_past_findings(query: str) -> str:
    """Recall bugs, quirks, and session history related to a query."""
    # Busca semanticamente en bugs, sesiones y quirks
    bugs = await store.asearch(namespace, query=query, limit=8)
    ...
```

Un agente puede invocar:
```json
{"name": "recall_past_findings", "args": {"query": "login page validation"}}
```

Y recibir:
```
KNOWN BUGS (2):
  - Save button returns 500 on concurrent requests (seen 3x, status: open)
  - Date picker overlaps with footer on mobile (seen 1x, status: open)

PAST SESSIONS (1):
  - test_settings_01: 15 actions, 2 bugs, outcome=bugs_found

KNOWN QUIRKS (1):
  - Settings page uses debounced save with 5s delay (confirmed 2x)
```

#### `record_observation` (Langmem `create_manage_memory_tool`)

Los agentes pueden registrar proactivamente observaciones de alto nivel que el procesamiento estructurado del Action Tape no captura:

```json
{"name": "record_observation", "args": {"content": "The settings page has an unusual 5-second debounce on save that causes confusion"}}
```

Estas observaciones se almacenan en el namespace `("app", url_hash, "agent_observations")` y aparecen en la seccion `AGENT_OBSERVATIONS` del `MEMORY_CONTEXT` del supervisor en ejecuciones futuras.

---

## 12. Integraciones Externas: MCP y Skills

### MCP: Model Context Protocol

#### Que es MCP?

El **Model Context Protocol** (MCP) es un protocolo abierto (donado a la Linux Foundation en diciembre 2025) que permite a los LLMs interactuar con herramientas externas de forma estandarizada. Fue creado originalmente por Anthropic.

**Analogia:** MCP es como un "USB para herramientas de IA" -- cualquier herramienta que implemente el protocolo puede conectarse a cualquier agente que lo soporte.

#### langchain-mcp-adapters

Este proyecto usa `langchain-mcp-adapters` para convertir herramientas MCP en herramientas LangChain:

```python
# custom_tools.py (simplificado)
from langchain_mcp_adapters.client import MultiServerMCPClient

async def get_mcp_tools(config_path: str) -> list:
    """Carga herramientas MCP desde la configuracion."""

    # 1. Leer configuracion de servidores MCP
    with open(config_path) as f:
        config = json.load(f)

    # 2. Conectar a todos los servidores
    async with MultiServerMCPClient(config["mcpServers"]) as client:
        tools = client.get_tools()

    return tools
```

#### Configuracion (mcp_servers.json)

```json
{
    "mcpServers": {
        "github": {
            "transport": "http",
            "url": "https://api.githubcopilot.com/mcp/"
        },
        "example-docs": {
            "transport": "http",
            "url": "https://example.com/docs/_mcp/"
        }
    }
}
```

Cada servidor MCP expone herramientas especificas. Por ejemplo, el servidor de GitHub expone:
- `get_pull_request` -- obtener datos de un PR
- `get_pull_request_files` -- archivos modificados
- `get_pull_request_diff` -- diff del PR

> **Documentacion oficial:** [github.com/langchain-ai/langchain-mcp-adapters](https://github.com/langchain-ai/langchain-mcp-adapters)

### Skills: Scripts Externos

El proyecto tambien soporta "skills" -- scripts externos que los agentes pueden ejecutar:

```python
# Buscar informacion de un skill
resultado = fetch_agent_skill("mi_skill")
# Retorna: SKILL.md + lista de scripts disponibles

# Ejecutar un script
resultado = run_agent_skill_script("mi_skill", "verificar_datos.py", args=["--check"])
# Ejecuta en subprocess con timeout de 60s
```

Los skills siguen un modelo de **progressive disclosure**: primero se muestra la documentacion (`SKILL.md`), luego los scripts disponibles, y solo se ejecutan cuando el agente los necesita.

---

## 13. Las Misiones: Formato YAML y Enrutamiento

### Formato de Mision

Una mision es un archivo YAML que describe que testear en lenguaje natural:

```yaml
# missions/new_user_agent.yaml
missions:
  - thread_id: "test_onboarding_01"
    prompt: >
      You are testing the application for the first time.
      Navigate to the login page, explore the main dashboard,
      and document any confusing UX patterns you find.
      Focus on discoverability of features.

  - thread_id: "test_onboarding_02"
    prompt: >
      Test the signup flow as a completely new user.
      Try creating an account with various input combinations.
```

### Enrutamiento: Standard vs. Advanced

El `thread_id` determina que grafo se usa (main.py:41):

```python
ADVANCED_KEYWORDS = (
    "accessibility", "a11y",
    "data_heavy", "data-heavy",
    "impatient",
    "returning",
    "explorer", "chaos", "autonomous",
)

# Si el thread_id contiene alguna de estas palabras -> grafo avanzado
# Si no -> grafo estandar (3 personas)
```

Ejemplo:
- `thread_id: "test_login_01"` -> **Grafo estandar** (3 agentes)
- `thread_id: "test_accessibility_forms"` -> **Grafo avanzado** (5 agentes + explorador)
- `thread_id: "chaos_navigation_stress"` -> **Grafo avanzado**

### Misiones de Regresion

El sistema puede generar misiones automaticamente desde el catalogo de bugs (memory.py:546):

```python
async def generate_regression_missions(store, url_hash):
    bugs = await store.asearch(("episodes", url_hash, "bugs"), limit=30)
    open_bugs = [b for b in bugs if b.value["status"] == "open"]

    missions = []
    for page, bug_summaries in pages_with_bugs.items():
        missions.append({
            "thread_id": f"regression_01_{page}",
            "prompt": f"Regression test for page '{page}'. "
                      f"Previously discovered bugs: {bug_summaries}. "
                      "Verify whether these bugs still exist."
        })
    return missions
```

Se activa con `--regression`:
```bash
agent-explorer --regression --headed
```

---

## 14. Generacion de Tests desde Pull Requests

### Flujo PR -> Misiones

El sistema puede analizar un Pull Request de GitHub y generar misiones de testing automaticamente:

```bash
agent-explorer --pr-url https://github.com/org/repo/pull/123 --execute
```

El flujo es:

```
   PR URL
     |
     v
  fetch_pr_data()         # MCP (GitHub) o `gh` CLI
     |
     v
  {title, body,           # Metadatos del PR
   files, diff}
     |
     v
  generate_missions_from_pr()   # LLM analiza y genera
     |
     v
  [Mission YAML]          # Misiones listas para ejecutar
     |
     v
  run_missions()          # Ejecucion normal del grafo
```

### Doble estrategia de obtencion de datos

El `pr_analyzer.py` intenta primero con MCP y luego cae a `gh` CLI:

```python
async def fetch_pr_data(pr_url, mcp_config_path):
    # 1. Intentar con MCP (GitHub server)
    try:
        return await _fetch_pr_data_mcp(owner, repo, pr_number, mcp_config)
    except Exception:
        pass

    # 2. Fallback: gh CLI
    return await _fetch_pr_data_gh(owner, repo, pr_number)
```

### Control de presupuesto de contexto

Los diffs de PRs pueden ser enormes. El sistema aplica presupuestos estrictos:

| Parametro | Default | Variable de Entorno |
|---|---|---|
| Diff maximo | 40 KB | `PR_PROMPT_DIFF_BUDGET_CHARS` |
| Body maximo | 8 KB | `PR_PROMPT_BODY_BUDGET_CHARS` |
| Lista de archivos | 80 files | `PR_PROMPT_FILE_LIST_LIMIT` |
| Prompt de mision | 1.2 KB | `PR_GENERATED_MISSION_PROMPT_MAX_CHARS` |

Cuando el diff excede el presupuesto, se aplica un algoritmo de "presupuesto por archivo" que distribuye el espacio entre los archivos modificados:

```python
def _build_diff_excerpt(diff_text, budget):
    """Smarter diff trimming:
    - Diffs pequenos: se incluyen completos
    - Diffs grandes: se asigna un presupuesto por archivo
    - Siempre se preservan los headers de archivo y las lineas cambiadas
    """
```

---

## 15. Flujo de Ejecucion Completo

### Diagrama secuencial paso a paso

```
1. INICIO
   |
   v
2. CLI parsea argumentos (--missions, --headed, --provider, etc.)
   |
   v
3. Carga config.yaml + .env (interpolacion de variables de entorno)
   |
   v
4. Auto-detecta proveedor LLM (Claude o Gemini)
   |
   v
5. Inicializa Playwright (browser, contexto, pagina)
   |
   v
6. Inicializa persistencia:
   |-- AsyncSqliteSaver (checkpoints.db)
   |-- AsyncSqliteStore (memory.db)
   |
   v
7. Carga herramientas:
   |-- MCP tools (mcp_servers.json)
   |-- Agent Skills (agent-skills/)
   |
   v
8. Para cada mision en el YAML:
   |
   |  8a. Determinar grafo (standard vs. advanced) por keywords
   |      |
   |  8b. build_graph() o build_advanced_graph():
   |      |-- Crear agentes con prompts personalizados
   |      |-- Cargar suplementos de memoria procedural
   |      |-- Compilar StateGraph con checkpointer + store
   |      |
   |  8c. Ejecutar con astream():
   |      |
   |      |  CICLO DEL GRAFO:
   |      |  +--> SUPERVISOR
   |      |  |    - Lee mision + progreso + bugs + memoria
   |      |  |    - Elige siguiente agente (o FINISH)
   |      |  |    - Controla limite de pasos
   |      |  |
   |      |  +--> AGENTE SELECCIONADO
   |      |  |    - Razona (AIMessage)
   |      |  |    - Llama herramientas (ToolMessage)
   |      |  |      - execute_browser_command -> JSON -> Playwright
   |      |  |      - get_dom_snapshot -> lee la pagina
   |      |  |      - capture_bug_screenshot -> evidencia visual
   |      |  |      - generate_reproduction_spec -> .spec.ts
   |      |  |      - recall_past_findings -> consulta memoria
   |      |  |    - Escribe memorias semanticas desde Action Tape
   |      |  |
   |      |  +--> SUMMARIZER
   |      |       - Si hay muchos mensajes, comprime los viejos
   |      |       - Mantiene: primer HumanMessage + 20 recientes
   |      |       - Reemplaza el medio con un SystemMessage resumen
   |      |
   |      |  (REPITE hasta FINISH o error)
   |      |
   |  8d. Generar reporte (LLM resume el transcript)
   |      |
   |  8e. Escribir memoria episodica:
   |      |-- Session summary
   |      |-- Bug catalog (deduplicado)
   |
   v
9. Post-batch:
   |-- Reflexion procedural (LLM analiza todas las sesiones)
   |-- Actualizar suplementos de agentes y reglas de routing
   |
   v
10. Artefactos generados por mision:
    report_{thread_id}/
    |-- traces.log              # Transcript completo
    |-- test_report.md          # Reporte ejecutivo generado por LLM
    |-- action_tape.jsonl       # Log inmutable de acciones
    |-- reproduction_*.spec.ts  # Scripts Playwright reproducibles
    |-- screenshots/            # Evidencia visual de bugs
```

### Manejo de errores transitorios

El sistema implementa **retry con backoff exponencial** para errores transitorios (main.py):

```python
def _is_transient_error(exc):
    """Detecta errores recuperables."""
    text = str(exc).lower()
    return any(k in text for k in ("rate limit", "429", "503", "overloaded"))

# En el loop de ejecucion:
for attempt in range(max_retries):
    try:
        async for snapshot in graph.astream(state, config):
            ...
    except Exception as exc:
        if _is_transient_error(exc) and attempt < max_retries - 1:
            wait = 2 ** attempt * 5  # 5s, 10s, 20s, 40s...
            await asyncio.sleep(wait)
            continue
        raise
```

---

## 16. Patrones Arquitectonicos Clave

### 1. Record-and-Translate

**Problema:** Los agentes de IA son no-deterministas. Si un agente interactua directamente con el navegador, no puedes reproducir un bug.

**Solucion:** Separar la **intencion** (JSON) de la **ejecucion** (Playwright). Todo se graba en un tape inmutable que luego se traduce a codigo ejecutable.

### 2. Supervisor con Structured Output

**Problema:** Un LLM que decide "a donde ir" en texto libre puede generar nombres de agentes inexistentes o ambiguos.

**Solucion:** Usar `with_structured_output(schema)` para forzar al LLM a responder con un JSON valido contra un enum de valores permitidos.

### 3. Bounded State con Reducers Custom

**Problema:** En un grafo ciclico, los mensajes y datos se acumulan indefinidamente, agotando el contexto del LLM.

**Solucion:** Usar reducers custom que limitan el tamano del estado:
- `_bounded_tape_reducer`: Maximo 50 entradas en el Action Tape
- `make_summarizer_node`: Comprime mensajes antiguos manteniendo el primero + 20 recientes

### 4. Fabrica de Herramientas con Closure

**Problema:** Las herramientas del agente necesitan acceso a recursos (la `page` de Playwright, el `store` de memoria) que no son parametros del agente.

**Solucion:** Funciones fabrica que capturan recursos via closure:

```python
def get_browser_command_tool(page: Page):  # page capturada en closure
    @tool
    async def execute_browser_command(command_json: str, config: RunnableConfig) -> str:
        await page.click(...)  # page disponible aqui
    return execute_browser_command
```

### 5. Selector Resilience Guard

**Problema:** Los selectores CSS fragiles (basados en posicion) se rompen con cualquier cambio en el DOM.

**Solucion:** Validacion en tiempo de ejecucion que rechaza selectores posicionales y guia al agente hacia alternativas estables.

### 6. Memoria Cognitiva Multi-Nivel

**Problema:** Los agentes no aprenden de sesiones pasadas.

**Solucion:** Cuatro niveles de memoria (semantica, episodica, procedural, priorizacion) que informan tanto a los agentes (via prompt supplements) como al supervisor (via routing context).

### 7. Context Budget Management

**Problema:** Los LLMs tienen ventana de contexto limitada. Enviar todo el historial, todos los diffs, y toda la memoria puede exceder el limite.

**Solucion:** Presupuestos estrictos en cada punto:
- Transcript del reporte: 35KB maximo (head + tail)
- Diff de PRs: 40KB con distribucion por archivo
- Mensajes del supervisor: compactados a una linea por mensaje
- Memoria: limitada a las N entradas mas relevantes

---

## 17. Glosario

| Termino | Definicion |
|---|---|
| **StateGraph** | Grafo de estados de LangGraph que define el flujo de trabajo |
| **Node (Nodo)** | Funcion que transforma el estado del grafo |
| **Edge (Arista)** | Conexion entre nodos que define el flujo de ejecucion |
| **Conditional Edge** | Arista cuyo destino depende del estado actual |
| **Reducer** | Funcion que define como se combinan actualizaciones de un campo del estado |
| **Checkpointer** | Backend que persiste snapshots del estado para reanudar ejecuciones |
| **Store** | Almacen key-value para memoria a largo plazo entre hilos |
| **Supervisor** | Nodo central que decide que agente actua siguiente |
| **Swarm** | Patron multi-agente con enrutamiento dinamico |
| **Action Tape** | Log inmutable de todas las acciones del navegador (JSONL) |
| **Record-and-Translate** | Patron que separa intencion (JSON) de ejecucion (Playwright) |
| **MCP** | Model Context Protocol -- protocolo para herramientas externas |
| **Tool** | Funcion que un LLM puede decidir llamar durante su razonamiento |
| **Structured Output** | Forzar al LLM a responder con JSON validado contra un esquema |
| **ReAct** | Reasoning + Acting -- ciclo de razonamiento y accion del agente |
| **Thread ID** | Identificador unico de una ejecucion/mision del grafo |
| **Namespace** | Ruta jerarquica para organizar datos en el Store |
| **DOM Snapshot** | Representacion compacta del estado visual de la pagina web |
| **Accessibility Tree** | Representacion del DOM orientada a lectores de pantalla |

---

## 18. Referencias y Documentacion Oficial

### LangGraph

| Recurso | URL |
|---|---|
| Documentacion principal | [docs.langchain.com/oss/python/langgraph/graph-api](https://docs.langchain.com/oss/python/langgraph/graph-api) |
| Referencia StateGraph | [reference.langchain.com/python/langgraph/graph/state/StateGraph](https://reference.langchain.com/python/langgraph/graph/state/StateGraph) |
| Persistencia (Checkpoints + Store) | [docs.langchain.com/oss/python/langgraph/persistence](https://docs.langchain.com/oss/python/langgraph/persistence) |
| AsyncSqliteSaver | [reference.langchain.com/python/langgraph.checkpoint.sqlite/aio/AsyncSqliteSaver](https://reference.langchain.com/python/langgraph.checkpoint.sqlite/aio/AsyncSqliteSaver) |
| Subgraphs | [docs.langchain.com/oss/python/langgraph/use-subgraphs](https://docs.langchain.com/oss/python/langgraph/use-subgraphs) |
| Human-in-the-loop | [docs.langchain.com/oss/python/langchain/human-in-the-loop](https://docs.langchain.com/oss/python/langchain/human-in-the-loop) |

### Patrones Multi-Agente

| Recurso | URL |
|---|---|
| langgraph-supervisor (PyPI) | [pypi.org/project/langgraph-supervisor](https://pypi.org/project/langgraph-supervisor/) |
| langgraph-supervisor (GitHub) | [github.com/langchain-ai/langgraph-supervisor-py](https://github.com/langchain-ai/langgraph-supervisor-py) |
| langgraph-swarm (PyPI) | [pypi.org/project/langgraph-swarm](https://pypi.org/project/langgraph-swarm/) |
| langgraph-swarm (GitHub) | [github.com/langchain-ai/langgraph-swarm-py](https://github.com/langchain-ai/langgraph-swarm-py) |

### LangChain

| Recurso | URL |
|---|---|
| Herramientas (Tools) | [docs.langchain.com/oss/python/langchain/tools](https://docs.langchain.com/oss/python/langchain/tools) |
| ChatAnthropic (Claude) | [docs.langchain.com/oss/python/integrations/chat/anthropic](https://docs.langchain.com/oss/python/integrations/chat/anthropic) |
| ChatGoogleGenerativeAI (Gemini) | [reference.langchain.com/python/integrations/langchain_google_genai/ChatGoogleGenerativeAI](https://reference.langchain.com/python/integrations/langchain_google_genai/ChatGoogleGenerativeAI/) |

### Integraciones

| Recurso | URL |
|---|---|
| langchain-mcp-adapters | [github.com/langchain-ai/langchain-mcp-adapters](https://github.com/langchain-ai/langchain-mcp-adapters) |
| MCP (Agentic AI Foundation) | [docs.langchain.com/oss/python/langchain/mcp](https://docs.langchain.com/oss/python/langchain/mcp) |

### Playwright

| Recurso | URL |
|---|---|
| Documentacion Python | [playwright.dev/python](https://playwright.dev/python/) |
| API Reference | [playwright.dev/python/docs/api/class-playwright](https://playwright.dev/python/docs/api/class-playwright) |
| Guia de la libreria | [playwright.dev/python/docs/library](https://playwright.dev/python/docs/library) |

### Proveedores de LLM

| Recurso | URL |
|---|---|
| Anthropic (Claude) API | [docs.anthropic.com](https://docs.anthropic.com) |
| Google AI (Gemini) | [ai.google.dev](https://ai.google.dev) |

---

> **Nota:** Este documento fue generado como material didactico y refleja el estado del proyecto y las APIs a fecha de mayo 2026. Las APIs de LangGraph y LangChain evolucionan rapidamente; consulta siempre la documentacion oficial para verificar la informacion mas actualizada.
