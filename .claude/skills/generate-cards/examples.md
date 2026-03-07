# Card Creation Examples by Domain

## Programming (Python)

```
Card 1 (basic):
  Front: What is a Python decorator?
  Back: A function that wraps another function to extend its behavior.
        Uses @syntax. Common uses: logging, timing, caching.
  Tags: python_functions, decorators, difficulty::medium

Card 2 (basic):
  Front: Why use @functools.wraps in a decorator?
  Back: Preserves the wrapped function's metadata (__name__, __doc__, etc.).
        Without it, introspection and debugging tools show the wrapper's info.
  Tags: python_functions, functools, difficulty::medium

Card 3 (cloze):
  Front: The {{c1::Global Interpreter Lock (GIL)}} prevents multiple threads
         from executing Python bytecode simultaneously.
  Tags: python_concurrency, gil, difficulty::hard
```

## Programming (Kotlin)

```
Card 1 (basic):
  Front: What is a Kotlin coroutine?
  Back: A lightweight thread that can be suspended without blocking.
        Uses `suspend` keyword. Runs in CoroutineScope.
  Tags: kotlin::coroutines, difficulty::medium

Card 2 (comparison):
  Front: What is the difference between launch and async in Kotlin?
  Back: - **launch**: Returns Job, "fire and forget", use when you don't need result
        - **async**: Returns Deferred<T>, use when you need to await a result
  Tags: kotlin::coroutines, difficulty::medium

Card 3 (cloze):
  Front: Use {{c1::StateFlow}} for state that needs to be observed,
         {{c2::SharedFlow}} for events.
  Tags: kotlin::flow, difficulty::hard
```

## Language Learning (Spanish)

```
Card 1 (reversed):
  Front: What is the Spanish word for "library" (building)?
  Back: biblioteca
  Tags: spanish_vocabulary, places, difficulty::easy

Card 2 (reversed):
  Front: biblioteca
  Back: library (building with books, NOT code library)
  Tags: spanish_vocabulary, places, difficulty::easy

Card 3 (basic):
  Front: When do you use "ser" vs "estar" for location?
  Back: - **estar**: Temporary location ("Estoy en casa" - I'm at home)
        - **ser**: Permanent location/events ("La fiesta es en mi casa")
  Tags: spanish_grammar, ser_vs_estar, difficulty::medium
```

## Science (Biology)

```
Card 1 (basic):
  Front: What is the function of mitochondria in a cell?
  Back: **Energy production** - converts glucose to ATP through cellular respiration.
        Often called the "powerhouse of the cell."
  Tags: biology_cell, organelles, difficulty::easy

Card 2 (basic):
  Front: Why do mitochondria have their own DNA?
  Back: Evidence of **endosymbiotic origin** - mitochondria were once free-living
        bacteria that entered a symbiotic relationship with ancestral cells.
  Tags: biology_cell, mitochondria, difficulty::medium

Card 3 (cloze):
  Front: The {{c1::inner membrane}} of mitochondria contains the enzymes
         for the {{c2::electron transport chain}}.
  Tags: biology_cell, mitochondria, difficulty::medium
```

## History

```
Card 1 (basic):
  Front: What were the main causes of World War I?
  Back: **MAIN** mnemonic:
        - **M**ilitarism - arms race between powers
        - **A**lliances - entangling defense pacts
        - **I**mperialism - competition for colonies
        - **N**ationalism - ethnic tensions
  Tags: history_wwi, causes, difficulty::medium

Card 2 (basic):
  Front: What event triggered the start of World War I?
  Back: Assassination of **Archduke Franz Ferdinand** of Austria-Hungary
        in Sarajevo on June 28, 1914 by Gavrilo Princip.
  Tags: history_wwi, trigger, difficulty::easy
```

## Mathematics

```
Card 1 (basic):
  Front: What is the derivative of sin(x)?
  Back: **cos(x)**
        Derivation: lim(h->0) [sin(x+h) - sin(x)]/h = cos(x)
  Tags: math_calculus, derivatives, difficulty::medium

Card 2 (cloze):
  Front: The derivative of {{c1::e^x}} is {{c2::e^x}}.
  Tags: math_calculus, derivatives, difficulty::easy

Card 3 (basic):
  Front: What is the chain rule for derivatives?
  Back: If y = f(g(x)), then dy/dx = **f'(g(x)) * g'(x)**
        "Derivative of outside times derivative of inside"
  Tags: math_calculus, chain_rule, difficulty::medium

Card 4 (cloze):
  Front: The integral of {{c1::1/x}} is {{c2::ln|x|}} + C
  Tags: math_calculus, integration, difficulty::medium
```

## Good vs Bad Examples

### Good Card

```
Front: Why doesn't Python threading improve CPU-bound performance?
Back: The **GIL** (Global Interpreter Lock) allows only one thread
      to execute Python bytecode at a time.
      Solutions:
      - Use multiprocessing for CPU parallelism
      - Threading still helps I/O-bound tasks
Tags: python_concurrency, gil, difficulty::hard
```

### Bad Card (Too Vague)

```
Front: Explain threading in Python
Back: [500 word essay]
```

**Fix**: Break into specific questions about GIL, threading vs multiprocessing, use cases.

### Bad Card (Set/Enumeration)

```
Front: List all Python data types
Back: int, float, str, list, tuple, dict, set, bool, None...
```

**Fix**: Split into separate cards for each type, or create comparison cards.
