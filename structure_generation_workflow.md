graph TD
    A["Start: Generate All Potential Configurations<br>(C<sub>1</sub>, C<sub>2</sub>, ..., C<sub>N</sub>)<br>e.g., using itertools.combinations"] --> B{"Loop: For each Potential Configuration C<sub>i</sub>"};

    B --> C["Apply Full Set of Symmetry Operations G<br>(e.g., 48 ops for O<sub>h</sub> group) to C<sub>i</sub>"];
    C --> D["Generate the Orbit of C<sub>i</sub>:<br>Consists of |G| transformed configurations<br>{g<sub>1</sub>(C<sub>i</sub>), g<sub>2</sub>(C<sub>i</sub>), ..., g<sub>|G|</sub>(C<sub>i</sub>)}"];

    D --> E{"Loop: For each Transformed Configuration C'<sub>i,j</sub> in the Orbit"};
    E --> F["1. Normalize Coordinates<br>(e.g., to [0,1) range, apply system-specific rules)"];
    F --> G["2. Sort Normalized Coordinates<br>(e.g., lexicographically by x, then y, then z)"];
    G --> H["Obtain Standardized Form S(C'<sub>i,j</sub>)"];
    H --> E;

    E -- "All C'<sub>i,j</sub> in orbit processed" --> I["Select Lexicographically Smallest Standardized Form<br>as the Canonical Representative for C<sub>i</sub>'s Orbit:<br>C<sub>i,canonical</sub> = min{S(C'<sub>i,1</sub>), ..., S(C'<sub>i,|G|</sub>)}"];

    I --> J{"Is C<sub>i,canonical</sub> already in the<br>Set of Unique Canonical Forms Found So Far?"};
    J -- No --> K["Add C<sub>i,canonical</sub> to Unique Set<br>Store Original C<sub>i</sub> (or C<sub>i,canonical</sub>)<br>as a Unique Structure Representative"];
    J -- Yes --> L["Discard C<sub>i</sub><br>(Symmetrically Equivalent to an Already Found Unique Structure)"];

    K --> B;
    L --> B;

    B -- "All Potential Configurations C<sub>i</sub> processed" --> M["End: Final Set of Unique<br>Structure Representatives Identified"];

    subgraph Legend
        direction LR
        Legend_Ci["C<sub>i</sub>: An initial (potential) atomic configuration"]
        Legend_g["g<sub>j</sub> ∈ G: A symmetry operation from the point group G"]
        Legend_Orbit["{g<sub>j</sub>(C<sub>i</sub>)}: The set of all configurations symmetrically equivalent to C<sub>i</sub> (its orbit)"]
        Legend_S_Cij["S(C'<sub>i,j</sub>): Standardized form (normalized & sorted) of a transformed configuration"]
        Legend_Ccanon["C<sub>i,canonical</sub>: The unique canonical representative chosen for the orbit of C<sub>i</sub>"]
    end

    classDef startEnd fill:#C9DAF8,stroke:#333,stroke-width:2px,font-weight:bold;
    classDef process fill:#E2F0D9,stroke:#333,stroke-width:1px;
    classDef decision fill:#FCE5CD,stroke:#333,stroke-width:2px;
    classDef unique fill:#D5E8D4,stroke:#277722,stroke-width:2px;
    classDef discard fill:#F8CECC,stroke:#A0302A,stroke-width:2px;
    classDef loop fill:#FFF2CC,stroke:#D6B656,stroke-width:1px;

    class A,M startEnd;
    class C,D,F,G,H,I process;
    class J decision;
    class K unique;
    class L discard;
    class B,E loop;
