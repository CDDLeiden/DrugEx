"""OpenEye ImplicitMillsDean color force field, ported to self-contained SMARTS.

PROVENANCE (verbatim source of every SMARTS below):
    ccr2_gen/oeye/current_apps/apps/openeye/data/rocs/3.9.0.1/rocs/ImplicitMillsDean.cff

That file declares 6 color INTERACTION types -- donor, acceptor, cation, anion,
rings, hydrophobe -- each ``attractive gaussian weight=1.0 radius=1.0`` (.cff lines
185-190). Each type's PATTERN lines (.cff lines 147-183) reference ``$macro`` names
bound in the DEFINE block (.cff lines 40-131). OpenEye's SMARTS dialect allows a
``$macro`` to stand for a previously-defined sub-SMARTS; RDKit/CDPKit do not. This
module reproduces the 6 types by expanding every ``$macro`` inline into a
self-contained recursive SMARTS ``$(...)`` so the patterns compile unchanged in both
RDKit (``Chem.MolFromSmarts``) and CDPKit (``Chem.parseSMARTS``).

Why port it: the open-source color channels do NOT match OpenEye out of the box.
RDKit's rdShapeAlign default color is a 5-pattern fallback with NO hydrophobe; RDKit's
BaseFeatures.fdef treats only AROMATIC atoms as "rings" (so saturated rings are missed)
and its LumpedHydrophobe under-fires on alkyl chains; CDPKit's DefaultPharmacophoreGenerator
uses its own perception. The CCR2 references are hydrophobe- and ring-heavy, so these gaps
bias the color score. Using OE's own atom typing closes the gap.

OPT-IN: this module is inert until a scorer is constructed with ``color_ff="oe_mills_dean"``.

pH / protonation semantics (verbatim from the .cff, NOT invented here):
    The ImplicitMillsDean variant is *implicit*: the acid/base macros it would use for
    explicit protonation states (``ACprimaryAmine``, ``DacidOH``, ...) are commented out
    in the DEFINE block, and ``CATamine``/``anion`` are written to match the NEUTRAL
    (un-ionized) forms. Hence ``cation`` fires on a neutral amine and ``anion`` fires on a
    neutral carboxylic/sulfonic/phosphonic acid -- and, because the heavy-atom skeleton is
    unchanged, on their (de)protonated forms too. This makes the color typing invariant to
    the input protonation state (verified in tests_oe_color_ff.ProtonationInvariance).

.cff macros NOT expanded here (because the .cff itself disables them -- they appear only in
commented-out DEFINE lines and never reach a live PATTERN): ACprimaryAmine, ACtertiaryAmine,
ACsecondaryAmine (L61/66/74), DacidOH (L95). They are intentionally omitted; no live pattern
references them. Everything that a live PATTERN transitively needs IS expanded.
"""
from __future__ import annotations

import re
from typing import Dict, List, Tuple

from rdkit import Chem

# ---------------------------------------------------------------------------
# DEFINE block (.cff lines 40-131), verbatim bodies keyed by macro name.
# A leading "$name" inside a value is an OE macro reference, expanded below.
# ---------------------------------------------------------------------------
_DEFINE: Dict[str, str] = {
    # degree, independent of explicit/implicit H (.cff L40-43)
    "hd1": "[X1H0,X2H1,X3H2,X4H3,X5H4,X6H5]",
    "hd2": "[X2H0,X3H1,X4H2,X5H3,X6H4]",
    "hd3": "[X3H0,X4H1,X5H2,X6H3]",
    "hd4": "[X3H0,X4H0,X5H1,X6H2]",
    # hydrophobic (.cff L47-50)
    "php": "[#6,#16&$hd2&!$(S=*),#35,#53;R0;!$(*~[!#1;!#6;!$([#16;$hd2])])]",
    "thp": "[$php;$hd1]",
    "hp": "[$php;!$hd1]",
    "ehp": "[$hp;!$(*([$hp])[$hp])]",
    # acceptors (.cff L54-80)
    "ACamine": "[N;!$(N*=[!#6]);!$(N~[!#6;!#1]);!$(Na);!$(N#*);!$(N=*)]",
    "ACphosphate": "[O;$hd1;$(O~P(~O)~O)]",
    "ACcarboxylate": "[O;$hd1;$(O[C;!$(*N)]=O),$(O=[C;!$(*N)][O;$hd1])]",
    "ACwater": "[OH2]",
    "AChet6N": "[nH0;X2;$(n1aaaaa1)]",
    "ACphosphinyl": "[O;$(O=P);!$(O=P~O)]",
    "ACsulphoxide": "[O;$(O=[S;!$(S(~O)~O);$(S([#6])[#6])])]",
    "AChet5N": "[nH0;X2;$(n1aaaa1)]",
    "ACthiocarbonyl": "[S;X1;$(S=[#6])]",
    "AChydroxyl": "[O;$hd1;$(O-[C;!$(C=*)])]",
    "ACsulphate": "[O;$hd1;$(O~S(~O)~O)]",
    "ACamide": "[O;$(O=[#6][#7]);!$(O=[#6]([#7])[#7,#8,#16])]",
    "ACcarbamate": "[O;$(O=[#6]([#7])[#8])]",
    "ACurea": "[O;$(O=[#6]([#7])[#7])]",
    "ACester": "[O;$(O=[#6][#8]*);!$(O=[#6]([#7,#8,#16])[#8]*)]",
    "ACnitrile": "[N;$hd1;$(N#C)]",
    "ACimine": "[N;!$hd3;$(N(=C)C),$(N=[#6]);!$(N=[#6][#7,#8;!$(*S=O)])]",
    "ACketone": "[O;$hd1;$(O=[#6;$([H2]),$([H1]-[#6]),$(*([#6])[#6])])]",
    "ACphenol": "[O;$hd1;$(Oa)]",
    "ACether": "[O;$(*([#6;!$(*=[O,S,N])])[#6;!$(*=[O,S,N])])]",
    "ACprimaryAniline": "[N;$(Na);$hd1]",
    "ACnitro": "[O;$hd1;$(O~N~[O;$hd1])]",
    "AChet5O": "[o;X2;$(o1cccc1),$(o1ccccc1);!$(*[#6]=O)]",
    "ACsulphone": "[O;$(O=[S;$(S(~O)(~O)([#6,#7])[#6])])]",
    "strongAcceptor": "[$ACphosphate,$ACcarboxylate,$ACwater,$AChet6N,$ACphosphinyl]",
    "moderateAcceptor": "[$ACsulphoxide,$AChet5N,$ACthiocarbonyl,$AChydroxyl,$ACsulphate,$ACamide,$ACcarbamate,$ACurea]",
    "weakAcceptor": "[$ACnitrile,$ACimine,$ACketone,$ACester,$ACphenol,$ACether,$ACprimaryAniline,$ACnitro,$AChet5O,$ACsulphone]",
    # donors (.cff L92-114)
    "Damine": "[N;!$(N*=[!#6]);!$(N~[!#6;!#1]);!$(Na);!$(N#*);!$(N=*)]",
    "Dhet5NH": "[nH;$(n1aaaa1),$(n1aaaaa1)]",
    "DNpH": "[NH,H2,H3;+]",
    "Dhydroxyl": "[OH1;$hd1;$(O-C);!$(OC=[O,N,S])]",
    "Dwater": "[OH2]",
    "DprimaryAmide": "[N;$hd1;$(NC=O),$(NS=O)]",
    "DanilineNH": "[NH1,NH2;$hd2;$(Nc);!$(NS(=O)=O)]",
    "DamidineNH": "[NH1,NH2;$(N~C~N),$(N~C(~N)~N)]",
    "DsecondaryAmide": "[#7;$hd2;$(*[#6,#16]=O);!$(N(a)S=O)]",
    "DanilineNH2": "[N;$hd1;$(Nc)]",
    "DhydraN": "[NH1,NH2,NH3;$hd1&$(NN[#6]),$hd2&$(N(N)[#6])]",
    "DimineNH": "[NH1;$(N=C)]",
    "DphenylOH": "[OH1;$(Oc)]",
    "DprimaryAmine": "[$Damine;$hd1]",
    "DsecondaryAmine": "[$Damine;$hd2]",
    "strongDonor": "[$Dhet5NH,$DNpH,$Dhydroxyl]",
    "moderateDonor": "[$Dwater,$DprimaryAmide,$DanilineNH,$DamidineNH,$DsecondaryAmide,$DanilineNH2]",
    "weakDonor": "[$DhydraN,$DimineNH,$DphenylOH,$DprimaryAmine,$DsecondaryAmine]",
    # anion intermediates (.cff L118-124)
    "negHet": "[#8,#16;$hd1]",
    "terminalHet": "[#7,#8,#16;$hd1]",
    "ANarylsulfonamide": "[N;$(N(a)S(=O)(=O)*)]",
    "ANmalonic": "[C;!$hd4;$(C(C=[O,S])C=[O,S])]",
    "ANarylthiol": "[S;$hd1;$(Sa)]",
    "ANhalideion": "[I,Br,Cl,F;!H0,-]",
    "ANhydroxylamine": "[O;$hd1;$(ON~C),$(O[n+]),$(O=n);!$(ONC=[S,O,N])]",
    # cation intermediates (.cff L128-131)
    "CATnonewN": "[#7;!$(NC=O);!$(NS(=O)=O)]",
    "CATguanidine": "[$CATnonewN]!:[#6](!:[$CATnonewN])!:[$CATnonewN]",
    "CATguanidineC": "[#6]~[$CATguanidine]",
    "CATamine": "[N;!$(N*=[!#6]);!$(N~[!#6;!#1]);!$(Na);!$(N=*);!$(N#*)]",
}

# PATTERN lines per type (.cff L147-183), bodies verbatim (still containing $macros).
_RAW_PATTERNS: Dict[str, List[str]] = {
    # rings (.cff L147-150): any ring of size 3..6 (NOT aromatic-only).
    "rings": [
        "[R]~1~[R]~[R]~[R]1",
        "[R]~1~[R]~[R]~[R]~[R]1",
        "[R]~1~[R]~[R]~[R]~[R]~[R]1",
        "[R]~1~[R]~[R]~[R]~[R]~[R]~[R]1",
    ],
    # hydrophobe (.cff L155-164): the 9-pattern thp/hp/ehp/php chain model.
    "hydrophobe": [
        "[$thp]~*(~[$thp])~[$thp]",
        "[$thp][!$(*(~[$thp])(~[$thp])~[$thp]);!$(*=[N,S,O])][$thp]",
        "[$thp;!$(*~*~[$thp]);$(*~[$php])]",
        "[$thp;#35,#53]",
        "[$ehp][$hp][$hp][$ehp]",
        "[$ehp]([$ehp])[$hp][$ehp]",
        "[$hp]([$ehp])([$ehp])[$ehp]",
        "[$ehp][$hp][$hp][$hp][$ehp]",
        "[$ehp][$hp][$hp;$(*[$hp][$hp][$ehp])]",
    ],
    # acceptor (.cff L168)
    "acceptor": ["[$strongAcceptor,$moderateAcceptor,$weakAcceptor]"],
    # donor (.cff L169)
    "donor": ["[$strongDonor,$moderateDonor,$weakDonor]"],
    # cation (.cff L174-177): guanidine, amidine, azole, neutral amine (pH-implicit).
    "cation": [
        "[$CATnonewN]!:[#6;!$(C(N)(N)N)](!:[$CATnonewN])!:[$CATnonewN]",
        "[$CATnonewN]!:[#6;!$([$CATguanidineC]);!$(C(N)N)]!:[$CATnonewN]",
        "n:1cncc1",
        "[$CATamine]",
    ],
    # anion (.cff L179-183): carboxylate, sulfonate, phosphonate, tetrazole, + misc acids
    # (pH-implicit: matches the NEUTRAL acids).
    "anion": [
        "[$negHet][#6X3]~[$terminalHet]",
        "[$negHet][#16X4](~[$terminalHet])~[$terminalHet]",
        "[$negHet][#15X4](=O)[$negHet,$terminalHet]",
        "[n;$hd2]1[n;$hd2][n;$hd2][n;$hd2]c1",
        "[$ANarylsulfonamide,$ANmalonic,$ANarylthiol,$ANhalideion,$ANhydroxylamine]",
    ],
}

# The 6 declared INTERACTION types (.cff L185-190); used to validate completeness.
_SIX_TYPES = ("donor", "acceptor", "cation", "anion", "rings", "hydrophobe")

_MACRO_REF = re.compile(r"\$([A-Za-z][A-Za-z0-9]*)")


def _expand(smarts: str, _stack: Tuple[str, ...] = ()) -> str:
    """Recursively replace every ``$macro`` in ``smarts`` with its inline recursive SMARTS.

    OE writes ``$name`` for "an atom/group matching macro ``name``"; the RDKit/CDPKit spelling
    of that inside a SMARTS is a recursive query ``$( <expanded body> )``. We therefore wrap
    each macro body in ``$( ... )`` on substitution. A bracketed-atom macro body like
    ``[X1H0,...]`` becomes ``$([X1H0,...])``; a multi-atom body like ``php`` becomes
    ``$([...])`` too -- RDKit accepts both. Cycles raise (the .cff DEFINE order is acyclic).
    """
    def _sub(match: "re.Match[str]") -> str:
        name = match.group(1)
        if name not in _DEFINE:
            raise KeyError(f"undefined OE macro ${name} (not in ImplicitMillsDean DEFINE block)")
        if name in _stack:
            raise RuntimeError(f"cyclic OE macro reference through ${name}")
        return "$(" + _expand(_DEFINE[name], _stack + (name,)) + ")"

    return _MACRO_REF.sub(_sub, smarts)


def _build_expanded() -> Dict[str, List[str]]:
    """Expand every PATTERN body and verify it compiles under RDKit. Raise on any failure."""
    out: Dict[str, List[str]] = {}
    for color_name, raw_list in _RAW_PATTERNS.items():
        expanded: List[str] = []
        for raw in raw_list:
            smt = _expand(raw)
            if Chem.MolFromSmarts(smt) is None:
                raise ValueError(
                    f"OE color SMARTS for {color_name!r} failed to compile: {raw} -> {smt}"
                )
            expanded.append(smt)
        out[color_name] = expanded
    missing = set(_SIX_TYPES) - set(out)
    if missing:
        raise ValueError(f"OE color FF is missing required types: {sorted(missing)}")
    return out


# Public: {color_name: [self-contained SMARTS, ...]} for the 6 OE color types.
# Precompiled-validated at import (every pattern is guaranteed MolFromSmarts-non-None).
OE_MILLS_DEAN_IMPLICIT: Dict[str, List[str]] = _build_expanded()

# Precompiled RDKit query mols, in the SAME order as OE_MILLS_DEAN_IMPLICIT, so the RDKit
# annotation path can substructure-match without recompiling per call.
OE_MILLS_DEAN_PATTERNS: Dict[str, List[Chem.Mol]] = {
    name: [Chem.MolFromSmarts(s) for s in smarts]
    for name, smarts in OE_MILLS_DEAN_IMPLICIT.items()
}


def _cdpkit_type_map() -> Dict[str, Tuple[int, int]]:
    """Map each OE color name -> (CDPKit FeatureType, FeatureGeometry).

    Color features are positional spheres in ROCS-style overlay, so geometry is SPHERE for
    all six. Returns plain ints (the CDPKit enum values) so the map is importable even when
    CDPKit is absent; the values equal CDPL.Pharm.FeatureType.* / FeatureGeometry.SPHERE.
    """
    try:
        import CDPL.Pharm as Pharm

        ft = Pharm.FeatureType
        sphere = int(Pharm.FeatureGeometry.SPHERE)
        return {
            "donor": (int(ft.H_BOND_DONOR), sphere),
            "acceptor": (int(ft.H_BOND_ACCEPTOR), sphere),
            "cation": (int(ft.POSITIVE_IONIZABLE), sphere),
            "anion": (int(ft.NEGATIVE_IONIZABLE), sphere),
            "rings": (int(ft.AROMATIC), sphere),
            "hydrophobe": (int(ft.HYDROPHOBIC), sphere),
        }
    except ImportError:
        # CDPKit FeatureType / FeatureGeometry integer values (stable enum, CDPKit 1.x):
        # H_BOND_DONOR=5, H_BOND_ACCEPTOR=6, POSITIVE_IONIZABLE=4, NEGATIVE_IONIZABLE=3,
        # AROMATIC=2, HYDROPHOBIC=1, FeatureGeometry.SPHERE=1.
        return {
            "donor": (5, 1),
            "acceptor": (6, 1),
            "cation": (4, 1),
            "anion": (3, 1),
            "rings": (2, 1),
            "hydrophobe": (1, 1),
        }


# Public: {color_name: (CDPKit FeatureType int, FeatureGeometry int)} -- 1:1 with the RDKit
# color names emitted into PUBCHEM_PHARMACOPHORE_FEATURES, so the two backends agree on types.
OE_TO_CDPKIT_TYPE: Dict[str, Tuple[int, int]] = _cdpkit_type_map()

# CDPKit atom label that marks a pattern atom as a feature POSITION reference. When every
# matched atom carries it, the CDPKit feature is placed at the centroid of the match (the OE
# convention of centering a color feature on the matched group). Required: a
# PatternBasedFeatureGenerator emits a feature with NO 3D coordinates unless at least one
# matched atom is a position reference, which then crashes the downstream Gaussian color shape.
_CDPKIT_POS_REF_FLAG = 2  # == CDPL.Pharm.PatternBasedFeatureGenerator.PatternAtomLabelFlag.POS_REF_ATOM_FLAG


def build_cdpkit_oe_pharm_generator():
    """Build a CDPKit ``DefaultPharmacophoreGenerator`` configured to perceive ONLY the 6 OE
    color types via the expanded ImplicitMillsDean SMARTS.

    All of CDPKit's own default feature perceptions are disabled, then for each OE color type a
    ``PatternBasedFeatureGenerator`` is installed whose include-patterns are the type's OE
    SMARTS (every matched atom flagged as a position reference so the feature is centered on the
    match). Pass the result to ``GaussianShapeGenerator.setPharmacophoreGenerator`` to colour the
    Gaussian shape with OE typing.

    Raises:
        ImportError: if CDPKit is not installed.
    """
    import CDPL.Chem as CDPLChem
    import CDPL.Pharm as Pharm

    gen = Pharm.DefaultPharmacophoreGenerator()
    # Disable every feature type CDPKit perceives by default; we re-enable only our six.
    for t in (
        Pharm.FeatureType.HYDROPHOBIC,
        Pharm.FeatureType.AROMATIC,
        Pharm.FeatureType.NEGATIVE_IONIZABLE,
        Pharm.FeatureType.POSITIVE_IONIZABLE,
        Pharm.FeatureType.H_BOND_DONOR,
        Pharm.FeatureType.H_BOND_ACCEPTOR,
        Pharm.FeatureType.HALOGEN_BOND_DONOR,
        Pharm.FeatureType.HALOGEN_BOND_ACCEPTOR,
        Pharm.FeatureType.EXCLUSION_VOLUME,
    ):
        gen.enableFeature(t, False)

    for color_name, smarts_list in OE_MILLS_DEAN_IMPLICIT.items():
        ftype, geom = OE_TO_CDPKIT_TYPE[color_name]
        feat_gen = Pharm.PatternBasedFeatureGenerator()
        for smt in smarts_list:
            patt = CDPLChem.parseSMARTS(smt)
            for atom in patt.atoms:  # flag all matched atoms as position references
                CDPLChem.setAtomMappingID(atom, _CDPKIT_POS_REF_FLAG)
            feat_gen.addIncludePattern(patt, ftype, 1.0, geom, 1.0)
        gen.setFeatureGenerator(ftype, feat_gen)
        gen.enableFeature(ftype, True)
    return gen
