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
        Efficiently generate sequences for the current node by applying its mutations
        to the parent's sequences and then passing the result to its children.

        Args:
            parent_sequences (dict): The dictionary of sequences (N, S, ORF1a, etc.)
                after applying all mutations up to the parent node.
        """
        if not proteins:
            proteins = list[str](self.mutations.keys())

        # initialize the current node sequences from the parent's already-mutated sequences.
        sequence_map = {protein: parent_sequences[protein] for protein in proteins}

        # apply mutations for each protein for *this node's branch*
        for protein, mutations in self.mutations.items():
            if protein not in proteins:
                continue
            # Convert the sequence to a list for mutable, O(1) character access
            prot_seq = list[Any](sequence_map[protein])

            # deletions require special handling as they change sequence length
            # so we process them after substitutions/insertions.
            deletions_to_apply = []

            for mutation in mutations:
                orig_prot, position, new_prot = mutation
                position = int(position) - 1  # Convert to 0-based indexing

                # deletion (new_prot is '-') or insertion (orig_prot is '-')
                if new_prot == "-":
                    deletions_to_apply.append(position)

                    # Apply sanity check for deletion against the current list
                    try:
                        assert prot_seq[position] == orig_prot
                    except Exception as e:
                        print(
                            f"Protein {protein}: Expected {orig_prot} at deletion position {position} (1-based: {position+1}), found {prot_seq[position]}."
                        )
                        raise e

                # Handle Substitution (e.g., A87T)
                elif orig_prot != "-":
                    # Sanity check for substitution
                    try:
                        assert prot_seq[position] == orig_prot
                    except Exception as e:
                        # Log error but don't strictly halt unless you need perfect alignment
                        print(
                            f"Protein {protein}: Expected {orig_prot} at substitution position {position} (1-based: {position+1}), found {prot_seq[position]}. Applying mutation anyway."
                        )

                        # print(f"Previous five: {prot_seq[position - 5: position]}")
                        # print(f"Next five: {prot_seq[position + 1: position + 6]}")
                    # Apply the substitution
                    prot_seq[position] = new_prot
                elif orig_prot == "-":
                    print(
                        "Whoopsy daisy that is a complex case that needs to be handled"
                    )

            # apply deletions
            # sort positions in reverse order to ensure index validity when deleting.
            deletions_to_apply.sort(reverse=True)
            for pos in deletions_to_apply:
                prot_seq.pop(pos)

            # convert back to string once
            sequence_map[protein] = "".join(prot_seq)

        # assign the final sequences to the current node
        self.n_sequence = sequence_map["N"]
        self.s_sequence = sequence_map["S"]

        # recursively call for children, passing this node's mutated sequences
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
