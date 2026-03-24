import networkx as nx
from .nlp_utils import compute_valence
from collections import defaultdict
import pandas as pd


################################################################################################
## SVO Operations.
################################################################################################


def filter_svo_dataframe_by_tea(df, TEA, WDW2=None):
    """
    Filters the SVO (Subject-Verb-Object) DataFrame to include only rows where 'TEA' and 'WDW2' match the provided arguments.

    This function specifically considers 'TEA' to be one of "Who", "Did", or "What". Synonyms of these terms are not taken into account.

    If `WDW2` is set to None , we consider all edges of TEA.



    Parameters:
    df (pd.DataFrame): The SVO DataFrame to filter.
    TEA: The value to match in the 'TEA' column. Accepted values are:
        - "Who"
        - "Did"
        - "What"
    WDW2: The value to match in the 'WDW2' column.
        - If `WDW2` is set to None , we consider all edges of TEA.



    Returns:
    pd.DataFrame: The filtered DataFrame.
    """

    # Define valid TEA values in lowercase for case-insensitive comparison
    valid_tea_values = {"who", "did", "what"}

    # Normalize inputs to lowercase
    if not isinstance(TEA, str):
        raise ValueError(f"TEA must be a string. Provided type: {type(TEA)}")
    TEA_normalized = TEA.lower()

    if TEA_normalized not in valid_tea_values:
        raise ValueError(f"TEA must be one of {valid_tea_values}. Provided: '{TEA}'")

    if WDW2 is not None:
        if not isinstance(WDW2, str):
            raise ValueError(
                f"WDW2 must be a string or None. Provided type: {type(WDW2)}"
            )
        WDW2_normalized = WDW2.lower()
        if WDW2_normalized not in valid_tea_values:
            raise ValueError(
                f"WDW2 must be one of {valid_tea_values} or None. Provided: '{WDW2}'"
            )
    else:
        WDW2_normalized = None

    # Apply filtering
    if WDW2_normalized is None:
        filtered_df = df[
            (
                (df["TEA"].str.lower() == TEA_normalized)
                | (df["WDW2"].str.lower() == TEA_normalized)
            )
            & (df["Semantic-Syntactic"] == 0)
        ]
    else:
        filtered_df = df[
            (df["TEA"].str.lower() == TEA_normalized)
            & (df["WDW2"].str.lower() == WDW2_normalized)
            & (df["Semantic-Syntactic"] == 0)
        ]

    return filtered_df


def merge_svo_dataframes(df_list):
    """
    Merges a list of DataFrames containing SVO data into a single DataFrame.
    The 'svo_id' column is updated to ensure unique values across the merged DataFrame.

    Args:
        df_list (list): A list of DataFrames containing SVO data

    Returns:
        pandas.DataFrame: A single DataFrame containing all SVO data
    """
    merged_df = pd.DataFrame()
    svo_id_offset = 0
    for df in df_list:
        df_copy = df.copy()
        # Ensure svo_id is numeric, keeping NaNs
        df_copy["svo_id"] = pd.to_numeric(df_copy["svo_id"], errors="coerce")
        # Convert to nullable integer type to allow NaNs
        df_copy["svo_id"] = df_copy["svo_id"].astype("Int64")
        # Increment IDs only for non-null values
        df_copy.loc[df_copy["svo_id"].notna(), "svo_id"] += svo_id_offset
        merged_df = pd.concat([merged_df, df_copy], ignore_index=True)
        # Update offset for the next DataFrame, ignoring NaNs
        max_id = df_copy["svo_id"].max()
        if pd.notna(max_id):
            svo_id_offset = max_id + 1
    return merged_df


def export_hypergraphs(df):
    """
    Extracts hypergraphs from the DataFrame where Semantic-Syntactic is 0,
    allowing duplicates across sentences but not within the same sentence.

    Args:
        df (pandas.DataFrame): DataFrame containing the hypergraph data

    Returns:
        list: A list of hypergraph strings, potentially with duplicates
    """
    # Assume we have a 'sentence_id' column in the DataFrame
    # If not, you'll need to add logic to identify sentence boundaries

    hypergraphs = []
    sentence_hypergraphs = defaultdict(set)

    # Filter rows where Semantic-Syntactic is 0
    filtered_df = df[df["Semantic-Syntactic"] == 0]

    # Group by sentence_id and collect hypergraphs
    for svo_id, group in filtered_df.groupby("svo_id"):
        for hypergraph in group["Hypergraph"]:
            if hypergraph != "N/A" and hypergraph not in sentence_hypergraphs[svo_id]:
                hypergraphs.append(hypergraph)
                sentence_hypergraphs[svo_id].add(hypergraph)

    return hypergraphs


def export_subj(df):
    """
    Extracts a set of all unique subjects from the DataFrame.
    Each element in the set is a tuple of (subject, valence).
    """
    # Filter rows where TEA is 'Who' (indicative of subjects)
    df_subjects = df[df["TEA"] == "Who"]
    # Get unique subjects from 'Node 1'
    subjects = df_subjects["Node 1"].unique()
    # Compute valence and create a set of tuples
    subject_tuples = set()
    for subj in subjects:
        valence = compute_valence(subj)
        subject_tuples.add((subj, valence))
    return subject_tuples


def export_obj(df):
    """
    Extracts a set of all unique objects from the DataFrame.
    Each element in the set is a tuple of (object, valence).
    """
    # Filter rows where WDW2 is 'What' (indicative of objects)
    df_objects = df[df["WDW2"] == "What"]
    # Get unique objects from 'Node 2'
    objects = df_objects["Node 2"].unique()
    # Compute valence and create a set of tuples
    object_tuples = set()
    for obj in objects:
        valence = compute_valence(obj)
        object_tuples.add((obj, valence))
    return object_tuples


def export_verb(df):
    """
    Extracts a set of all unique verbs from the DataFrame.
    Each element in the set is a tuple of (verb, valence).
    """
    # Verbs can be in 'Node 1' where TEA is 'Did' or in 'Node 2' where WDW2 is 'Did'
    verbs_node1 = df[df["TEA"] == "Did"]["Node 1"]
    verbs_node2 = df[df["WDW2"] == "Did"]["Node 2"]
    # Combine and get unique verbs
    verbs = pd.concat([verbs_node1, verbs_node2]).unique()
    # Compute valence and create a set of tuples
    verb_tuples = set()
    for verb in verbs:
        valence = compute_valence(verb)
        verb_tuples.add((verb, valence))
    return verb_tuples


def add_node_with_type(G, node_id, label, node_type):
    """
    Add a node to the graph with a specific type.
    If the node already exists, update its type to include the new type.
    """
    if G.has_node(node_id):
        if "type" in G.nodes[node_id]:
            G.nodes[node_id]["type"].add(node_type)
        else:
            G.nodes[node_id]["type"] = set([node_type])
    else:
        G.add_node(node_id, type=set([node_type]), label=label)


def filter_subjects(df, subject_term):
    """
    Filters the DataFrame to keep rows where the 'svo_id' corresponds to entries
    where 'Node 1' contains the subject term and 'TEA' indicates a subject ('Who').

    Parameters:
    - df: The input DataFrame.
    - subject_term: The term to search for in subjects.

    Returns:
    - A filtered DataFrame containing only rows with matching 'svo_id's.
    """
    # Identify 'svo_id's where 'Node 1' contains the subject term and 'TEA' is 'Who'
    subject_svo_ids = set(
        df[
            (df["Node 1"].str.contains(subject_term, case=False, na=False))
            & (df["TEA"] == "Who")
        ]["svo_id"].unique()
    )

    # Filter the DataFrame to only include rows with these 'svo_id's
    filtered_df = df[df["svo_id"].isin(subject_svo_ids)]
    return filtered_df


def filter_objects(df, object_term):
    """
    Filters the DataFrame to keep rows where the 'svo_id' corresponds to entries
    where 'Node 2' contains the object term and 'TEA' indicates an object ('What').

    Parameters:
    - df: The input DataFrame.
    - subject_term: The term to search for in subjects.

    Returns:
    - A filtered DataFrame containing only rows with matching 'svo_id's.
    """
    # Identify 'svo_id's where 'Node 12' contains the subject term and 'TEA' is 'What'
    object_svo_ids = set(
        df[
            (df["Node 2"].str.contains(object_term, case=False, na=False))
            & (df["TEA"] == "What")
        ]["svo_id"].unique()
    )
    # Filter the DataFrame to only include rows with these 'svo_id's
    filtered_df = df[df["svo_id"].isin(object_svo_ids)]
    return filtered_df


def svo_to_graph(df, subject_filter=None, object_filter=None):
    """
    Convert a pandas DataFrame of SVO data into a graph.
    Optionally filters the data based on the subject_filter.
    """
    G = nx.Graph()

    if subject_filter is not None:
        df = filter_subjects(df, subject_filter)

    if object_filter is not None:
        df = filter_objects(df, object_filter)

    for index, row in df.iterrows():
        node1 = row["Node 1"]
        wdw1 = row["TEA"]
        node2 = row["Node 2"]
        wdw2 = row["WDW2"]
        hypergraph = row["Hypergraph"]
        sem_synt = row["Semantic-Syntactic"]

        # Determine node types based on TEA and WDW2
        node1_type = (
            "subject" if wdw1 == "Who" else "verb" if wdw1 == "Did" else "object"
        )
        node2_type = (
            "subject" if wdw2 == "Who" else "verb" if wdw2 == "Did" else "object"
        )

        # Create unique node IDs to differentiate between subjects, verbs, and objects
        node1_id = node1 + "_" + node1_type[0]
        node2_id = node2 + "_" + node2_type[0]

        # Add nodes with labels and types
        add_node_with_type(G, node1_id, label=node1, node_type=node1_type)
        add_node_with_type(G, node2_id, label=node2, node_type=node2_type)

        # Determine relation type based on 'Semantic-Syntactic' column
        relation_type = "synonym" if sem_synt == 1 else "syntactic"

        if G.has_edge(node1_id, node2_id):
            # Edge exists, update the 'relation' attribute
            existing_relation = G.edges[node1_id, node2_id].get("relation", set())
            if isinstance(existing_relation, str):
                existing_relation = set([existing_relation])
            existing_relation.add(relation_type)
            G.edges[node1_id, node2_id]["relation"] = existing_relation
            # Also update hypergraph
            existing_hypergraph = G.edges[node1_id, node2_id].get("hypergraph", set())
            if isinstance(existing_hypergraph, str):
                existing_hypergraph = set([existing_hypergraph])
            existing_hypergraph.add(hypergraph)
            G.edges[node1_id, node2_id]["hypergraph"] = existing_hypergraph
            # Increment weight for syntactic relations
            if relation_type == "syntactic":
                G.edges[node1_id, node2_id]["weight"] += 1
        else:
            # Initialize weight to 1 for syntactic relations
            weight = 1 if relation_type == "syntactic" else 0
            G.add_edge(
                node1_id,
                node2_id,
                relation=set([relation_type]),
                hypergraph=set([hypergraph]),
                weight=weight,
            )

    return G


################################################################################################
## Centrality measures.
################################################################################################


def wdw_weighted_degree_centrality(
    df,
    TEA,
    WDW2=None,
    remove_same_type=False,
    remove_node_type=None,
):
    """
    Filters the SVO (Subject-Verb-Object) DataFrame to include only rows where 'TEA' and 'WDW2' match the provided arguments.

    This function specifically considers 'TEA' to be one of "Who", "Did", or "What". Synonyms of these terms are not taken into account.

    If `WDW2` is set to None , we consider all edges of TEA.



    Parameters:
    df (pd.DataFrame): The SVO DataFrame to filter.
    TEA: The value to match in the 'TEA' column. Accepted values are:
        - "Who"
        - "Did"
        - "What"
    WDW2: The value to match in the 'WDW2' column.
        - If `WDW2` is set to None , we consider all edges of TEA.

    remove_same_type (bool): If True, removes edges between nodes of the same type (what-what, who-who).
    remove_node_type (str): If set to "What" or "Who", removes all rows containing that node type.

    Returns:
    pd.DataFrame: The filtered DataFrame.
    """

    # First filter the DataFrame based on node_type parameter
    if remove_node_type in ["What", "Who"]:
        df = df[
            ~df["TEA"].isin([remove_node_type]) & ~df["WDW2"].isin([remove_node_type])
        ]

    filtered_df = filter_svo_dataframe_by_tea(df, TEA, WDW2)

    if remove_same_type:
        # Keep only rows where TEA and WDW2 are different
        filtered_df = filtered_df[filtered_df["TEA"] != filtered_df["WDW2"]]

    G = svo_to_graph(filtered_df)

    # Compute weighted degrees (node strengths)
    node_strength = dict(G.degree(weight="weight"))
    # Normalize by total strength to get centrality
    total_strength = sum(node_strength.values())
    weighted_degree_centrality = {
        node: strength / total_strength for node, strength in node_strength.items()
    }
    # Create a DataFrame with 'node' as a column
    df = pd.DataFrame(
        list(weighted_degree_centrality.items()), columns=["node", "degree_centrality"]
    )

    # Filter the DataFrame to include only subjects, verbs, or objects
    if TEA == "Who":
        df_filtered = df[df["node"].str.endswith("_s")].copy()
    elif TEA == "Did" and WDW2 == None:
        df_filtered = df[df["node"].str.endswith("_v")].copy()
    elif WDW2 == "What":
        df_filtered = df[df["node"].str.endswith("_o")].copy()
    # Remove the last character(s) from the node names
    df_filtered["node"] = df_filtered["node"].str.replace("(_s|_v|_o)$", "", regex=True)

    # Sort the DataFrame
    df_sorted = df_filtered.sort_values(
        by="degree_centrality", ascending=False
    ).reset_index(drop=True)
    return df_sorted


def wdw_degree_centrality_overview(df):
    """
    Compute the degree centrality for each of the SVO components (Subject, Verb, Object) using the weighted degree centrality measure.
    """

    combinations = [
        ["Subject", "Who", "Did"],
        ["Verb", "Did", None],
        ["Object", "Did", "What"],
    ]

    for svo, TEA, WDW2 in combinations:

        print(f"Degree centrality for {svo}")
        display(wdw_weighted_degree_centrality(df, TEA, WDW2).head(20))
        print("############################################ \n")
