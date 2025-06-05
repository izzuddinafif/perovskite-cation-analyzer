graph TD
    A["Start: Generate All Potential Configurations<br>(C<sub>1</sub>, C<sub>2</sub>, ..., C<sub>N</sub>)<br>e.g., using itertools.combinations"] --> B["STEP 1: Apply Burnside's Lemma<br>|X/G| = (1/|G|) × Σ|X<sup>g</sup>|"];

    B --> C{"Loop: For each Symmetry Operation g ∈ G<br>(e.g., 48 operations for O<sub>h</sub> group)"};
    C --> D["Count Fixed Points |X<sup>g</sup>|:<br>Configurations unchanged by operation g"];
    D --> E{"Loop: For each Configuration C<sub>i</sub>"};
    E --> F["Apply g to C<sub>i</sub>: g(C<sub>i</sub>)"];
    F --> G{"Is g(C<sub>i</sub>) = C<sub>i</sub>?<br>(Configuration fixed by g?)"};
    G -- Yes --> H["Increment Fixed Count"];
    G -- No --> I["Continue to Next Configuration"];
    H --> E;
    I --> E;
    E -- "All C<sub>i</sub> processed" --> J["Store |X<sup>g</sup>| for this operation"];
    J --> C;

    C -- "All g ∈ G processed" --> K["Calculate: Total Fixed = Σ|X<sup>g</sup>|<br>Unique Count = Total Fixed / |G|"];

    K --> L["STEP 2: Generate Representative Configurations<br>Using Canonical Form Enumeration"];

    L --> M{"Loop: For each Potential Configuration C<sub>i</sub>"};
    M --> N["Apply Full Set of Symmetry Operations G to C<sub>i</sub>"];
    N --> O["Generate Orbit: {g<sub>1</sub>(C<sub>i</sub>), g<sub>2</sub>(C<sub>i</sub>), ..., g<sub>|G|</sub>(C<sub>i</sub>)}"];
    O --> P["For each transformed configuration:<br>1. Normalize coordinates<br>2. Sort lexicographically"];
    P --> Q["Select lexicographically smallest<br>as canonical representative C<sub>i,canonical</sub>"];

    Q --> R{"Is C<sub>i,canonical</sub> already in<br>unique representatives set?"};
    R -- No --> S["Add to unique set"];
    R -- Yes --> T["Discard (duplicate)"];
    S --> M;
    T --> M;

    M -- "All C<sub>i</sub> processed" --> U["STEP 3: Verification"];
    U --> V{"Does |Generated Representatives|<br>= Burnside's Lemma Count?"};
    V -- Yes --> W["✓ Success: Generate CIF Files"];
    V -- No --> X["✗ Error: Implementation Issue"];

    W --> Y["End: Unique Structures Generated"];
    X --> Z["End: Verification Failed"];

    subgraph "Burnside's Lemma Formula"
        direction TB
        BL1["|X/G| = Number of Unique Orbits"]
        BL2["X<sup>g</sup> = Configurations fixed by operation g"]
        BL3["G = Group of symmetry operations"]
        BL4["Formula: |X/G| = (1/|G|) × Σ<sub>g∈G</sub>|X<sup>g</sup>|"]
    end

    subgraph Legend
        direction LR
        Legend_Xi["X<sup>g</sup>: Set of configurations fixed by operation g"]
        Legend_g["g ∈ G: Symmetry operation from point group"]
        Legend_Canonical["C<sub>canonical</sub>: Lexicographically smallest representative"]
        Legend_Orbit["Orbit: All symmetrically equivalent configurations"]
    end

    classDef startEnd fill:#C9DAF8,stroke:#333,stroke-width:2px,font-weight:bold;
    classDef burnside fill:#E1D5E7,stroke:#9673A6,stroke-width:2px,font-weight:bold;
    classDef canonical fill:#E2F0D9,stroke:#333,stroke-width:1px;
    classDef decision fill:#FCE5CD,stroke:#333,stroke-width:2px;
    classDef success fill:#D5E8D4,stroke:#277722,stroke-width:2px;
    classDef error fill:#F8CECC,stroke:#A0302A,stroke-width:2px;
    classDef verification fill:#FFF2CC,stroke:#D6B656,stroke-width:2px;

    class A,Y,Z startEnd;
    class B,C,D,K burnside;
    class L,M,N,O,P,Q canonical;
    class G,R,V decision;
    class W success;
    class X error;
    class U verification;