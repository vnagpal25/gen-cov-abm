import re
import logging
from pathlib import Path
from typing import Any, List

# Set up logger for genome processing errors
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

# Create file handler that writes to genome-errors.log in the project root
log_file = Path(__file__).parent.parent.parent / "genome-errors.log"
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.WARNING)

# Create formatter and add it to the handler
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
file_handler.setFormatter(formatter)

# Add handler to logger (only if not already added to avoid duplicates)
if not logger.handlers:
    logger.addHandler(file_handler)


class CovidGenomeSequence:
    """
    Represents a COVID-19 genome sequence from Nextstrain data.
    All attributes are stored as primitives for easy CSV export.
    Note: mutations dict is kept for programmatic access but won't be exported to CSV.
    """

    # Class-level regex pattern shared across all instances
    # Matches mutations like: Q29*, D614G, del69-70 (where * = stop codon)
    _mut_pattern = re.compile(r"([A-Z]|\*)(\d+)([A-Z]|-|\*)")

    def __init__(self, nextstrain_node):
        self.name = nextstrain_node["name"]

        self.extract_node_info(nextstrain_node["node_attrs"])
        self.extract_mutations(nextstrain_node["branch_attrs"]["mutations"])

        # nextstrain's way of marking inferred ancestors
        if "NODE" in self.name:
            children = nextstrain_node.get("children")
            self.children = [self.__class__(child) for child in children]
        else:
            self.children = []

    def extract_node_info(self, node_attrs):
        """
        Extract all node attributes as primitive types (str, int, float, bool)
        for easy CSV export.
        """
        # Clade membership - tracks which strain/lineage a sample belongs to
        if "clade_membership" in node_attrs:
            self.clade_membership = node_attrs["clade_membership"]["value"]
        else:
            self.clade_membership = ""

        # Divergence - tracks how different the sequence is from the root
        if "div" in node_attrs:
            self.divergence = node_attrs["div"]
        else:
            self.divergence = ""

        # Division - location within state/country
        if "division" in node_attrs:
            self.division = node_attrs["division"]["value"]
        else:
            self.division = ""

        # Numeric date - collection/emergence date
        if "num_date" in node_attrs:
            self.num_date = node_attrs["num_date"]["value"]
        else:
            self.num_date = ""

        # Nextstrain clade classification
        if "Nextstrain_clade" in node_attrs:
            self.nextstrain_clade = node_attrs["Nextstrain_clade"]["value"]
        else:
            self.nextstrain_clade = ""

        # Author information
        if "author" in node_attrs:
            self.author = node_attrs["author"]["value"]
        else:
            self.author = ""

        # Country of collection
        if "country" in node_attrs:
            self.country = node_attrs["country"]["value"]
        else:
            self.country = ""

        # Country exposure
        if "country_exposure" in node_attrs:
            self.country_exposure = node_attrs["country_exposure"]["value"]
        else:
            self.country_exposure = ""

        # Division exposure
        if "division_exposure" in node_attrs:
            self.division_exposure = node_attrs["division_exposure"]["value"]
        else:
            self.division_exposure = ""

        # Geographic location (Northeast USA specific)
        if "geoloc_neusa" in node_attrs:
            self.geoloc_neusa = node_attrs["geoloc_neusa"]["value"]
        else:
            self.geoloc_neusa = ""

        # Host organism
        if "host" in node_attrs:
            self.host = node_attrs["host"]["value"]
        else:
            self.host = ""

        # Pango lineage classification
        if "pango_lineage" in node_attrs:
            self.pango_lineage = node_attrs["pango_lineage"]["value"]
        else:
            self.pango_lineage = ""

        # Region of collection (broad continental tagging)
        if "region" in node_attrs:
            self.region = node_attrs["region"]["value"]
        else:
            self.region = ""

        # Region exposure
        if "region_exposure" in node_attrs:
            self.region_exposure = node_attrs["region_exposure"]["value"]
        else:
            self.region_exposure = ""

        # URL to sequence data
        if "url" in node_attrs:
            self.url = node_attrs["url"]
        else:
            self.url = ""

        # Location (sub-regional)
        if "location" in node_attrs:
            self.location = node_attrs["location"]["value"]
        else:
            self.location = ""

        # Originating lab (Northeast USA specific attribute)
        if "Northeast_USA_originating_lab" in node_attrs:
            self.originating_lab = node_attrs["Northeast_USA_originating_lab"]["value"]
        else:
            self.originating_lab = ""

        # Submitting lab (Northeast USA specific attribute)
        if "Northeast_USA_submitting_lab" in node_attrs:
            self.submitting_lab = node_attrs["Northeast_USA_submitting_lab"]["value"]
        else:
            self.submitting_lab = ""

        # Purpose of sequencing
        if "purpose_of_sequencing" in node_attrs:
            self.purpose_of_sequencing = node_attrs["purpose_of_sequencing"]["value"]
        else:
            self.purpose_of_sequencing = ""

    def extract_mutations(self, mutations: dict):
        """
        Extract mutations from branch attributes.
        Mutations are stored as a dict mapping protein/gene to list of mutation tuples.
        Note: This dict won't be exported to CSV due to being non-primitive.
        """
        self.mutations = {}
        # mut_loc can be nuc (full nucleotide base mutations) or a protein type (S, E, M, N, ORF1a, etc.)
        for mut_loc, muts in mutations.items():
            mutations = []
            for mut in muts:
                if mut.startswith("-"):
                    logger.warning(f"{mut_loc}")

                    continue

                parsed = self._mut_pattern.findall(mut)[0]

                mutations.append(parsed)

            self.mutations[mut_loc] = mutations

    def generate_sequences(self, parent_sequences, proteins: List[str] = None):
        """
        Efficiently generate sequences by maintaining correct reference coordinates.

        Internal storage (parent_sequences) uses lists of strings where:
          - Deletions are marked as '-' (maintaining index alignment)
          - Insertions are appended to the residue at the site (maintaining index alignment)
        """
        if not proteins:
            proteins = list[str](self.mutations.keys())

        # initialize seq map from parent
        # deep copy because we are modifying lists in place
        sequence_map = {
            p: list(parent_sequences[p]) if p in parent_sequences else []
            for p in proteins
        }

        # Apply mutations
        for protein, mutations in self.mutations.items():
            if protein not in proteins:
                continue

            prot_seq = sequence_map[protein]

            for mutation in mutations:
                orig_prot, position, new_prot = mutation
                position = int(position) - 1  # 0-based index

                # ---- sanity check ----
                # want to make sure that the previous protein matches
                # what the mutation expects it's changing
                try:

                    current_val = prot_seq[position]

                    # if len(current_val) > 1:
                    #     print('multiple proteins here')

                    if len(current_val) > 1:
                        current_base = current_val[0]
                    else:
                        current_base = current_val
                    assert current_base == orig_prot
                except AssertionError:
                    print(
                        f"Warning {self.name}: {protein} pos {position+1} expected {orig_prot} found {current_base}"
                    )

                # handle deletion (X, pos, -)
                if new_prot == "-":
                    # replace with gap character
                    prot_seq[position] = "-"
                # handle insertion (-, pos, X)
                elif orig_prot == "-":
                    # append to previous residue to maintain indices.
                    if position < len(prot_seq):
                        prot_seq[position] += new_prot
                    else:
                        print(
                            f"Warning: {new_prot} inserted outside of reference frame"
                        )
                        prot_seq.append(new_prot)
                # handle substitution (X, Y)
                else:
                    # Simple overwrite
                    # If there was an existing insertion here, we might need to preserve it
                    # but usually substitutions replace the 'base' amino acid.
                    if len(prot_seq[position]) == 1:
                        # Keep the insertion tail, change the head
                        prot_seq[position] = new_prot

                    else:
                        print(
                            f"Warning: Substituting {new_prot} at previous insertion spot"
                        )
                        prot_seq[position] = new_prot + prot_seq[position][1:]

            sequence_map[protein] = prot_seq

        # remove gap characters
        self.n_sequence = "".join(sequence_map.get("N", "")).replace("-", "")
        self.s_sequence = "".join(sequence_map.get("S", "")).replace("-", "")
        self.orf1a_sequence = "".join(sequence_map.get("ORF1a", "")).replace("-", "")

        # pass aligned mutated proteins to children for further mutation
        for child in self.children:
            child.generate_sequences(sequence_map, proteins)

    def is_collected_tip(self) -> bool:
        """Check if this node is a collected sample (tip) or an inferred ancestor."""
        return "NODE" not in self.name

    def __str__(self):
        return f"COVID Genome sequence: {self.name}"

    def __eq__(self, other):
        return self.name == other.name

    def __hash__(self):
        return hash(self.name)
