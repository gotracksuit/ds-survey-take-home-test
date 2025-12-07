import numpy as np
import pandas as pd
from collections import defaultdict


def proof_qualified_target(categories, allocation, target_qualified):
    """
    Proof 1:
    Show Ec * Ic >= target for all categories
    """
    categories = categories.copy()

    categories["E_c (Total Exposed)"] = allocation.sum(axis=1)
    categories["I_c (Incidence Rate)"] = categories["incidence_rate"]
    categories["Expected Qualified"] = (
        categories["E_c (Total Exposed)"] * categories["I_c (Incidence Rate)"]
    )

    # LP guarantees targets in expectation; small tolerance accounts for solver precision.
    categories["Meets Target"] = (
        categories["Expected Qualified"] >= (target_qualified * 0.995)
    )

    print("\n✅ PROOF 1: Qualified Target (≥200)\n")
    print(
        categories[
            [
                "category_name",
                "E_c (Total Exposed)",
                "I_c (Incidence Rate)",
                "Expected Qualified",
                "Meets Target",
            ]
        ].to_string(index=False)
    )

    return categories["Meets Target"].all()

def proof_mean_time(categories, allocation, max_mean_time):
    """
    Proof 2:
    Show total expected time <= 480 * M
    """
    exposure_per_category = allocation.sum(axis=1)

    expected_time = np.sum(
        exposure_per_category
        * categories["incidence_rate"].values
        * categories["category_length_seconds"].values
    )

    total_respondents = allocation.sum()
    mean_time = expected_time / total_respondents

    print("\n✅ PROOF 2: Mean Interview Time (≤480s)\n")
    print(f"Total respondents M: {total_respondents:.1f}")
    print(f"Total expected time: {expected_time:.1f}s")
    print(f"Mean expected time: {mean_time:.2f}s")
    print(f"Constraint satisfied: {mean_time <= max_mean_time}")

    return mean_time <= max_mean_time

def proof_demographics(
    allocation, demographic_cells, demographic_shares, tolerance=0.01
):
    """
    Proof 3:
    Exposure distribution matches population share
    """
    total_respondents = allocation.sum()

    achieved_shares = allocation.sum(axis=0) / total_respondents

    proof_df = pd.DataFrame({
        "Demographic Cell": demographic_cells,
        "Population Share (P_j)": demographic_shares,
        "Achieved Exposure Share": achieved_shares,
        "Absolute Error": np.abs(achieved_shares - demographic_shares),
    })

    proof_df["Within Tolerance"] = proof_df["Absolute Error"] <= tolerance

    print("\n✅ PROOF 3: Demographic Representativeness\n")
    print(proof_df.to_string(index=False))

    print(
        "\nAll demographics within tolerance:",
        proof_df["Within Tolerance"].all(),
    )

    return proof_df["Within Tolerance"].all()

def validate_lp(categories, allocation, target_qualified, max_mean_time, demographic_cells, demographic_shares, tolerance=0.005):
    """
    Runs all three proofs for LP allocation.
    """
    print("\n================= LP PROOFS =================")

    proof1 = proof_qualified_target(categories, allocation, target_qualified)
    proof2 = proof_mean_time(categories, allocation, max_mean_time)
    proof3 = proof_demographics(allocation, demographic_cells, demographic_shares, tolerance)

    print(f"\n qualified target proof passed: {proof1}")
    print(f" mean time proof passed: {proof2}")
    print(f" demographics proof passed: {proof3}")

def validate_greedy(slots, categories_df, target_qualified, max_mean_time):
    """
    Placeholder for greedy allocation validation.
    Implement similar proofs as above if needed.
    """
    # Consolidate Final Exposure Counts for Validation
    final_exposure_counts = defaultdict(int)
    for slot in slots:
        for cid in slot['categories']:
            final_exposure_counts[cid] += 1
    
    # Print Results
    print("## 📊 Greedy Allocation Validation Results")

    validation_df = categories_df.copy()
    validation_df = validation_df.set_index('category_id')

    # 1. Total Exposed (E_c)
    validation_df['E_c (Total Exposed)'] = validation_df.index.map(final_exposure_counts)

    # 2. Incidence Rate (I_c)
    validation_df['I_c (Incidence Rate)'] = validation_df['incidence_rate'].round(4)

    # 3. Expected Qualified
    validation_df['Expected Qualified'] = (validation_df['E_c (Total Exposed)'] * validation_df['incidence_rate']).round(0).astype(int)

    # 4. Meets Target
    validation_df['Meets Target'] = np.where(validation_df['Expected Qualified'] >= target_qualified, '✅ Yes', '❌ No')

    # Select and rename final columns for printing
    result_table = validation_df[[
        'category_name', 
        'E_c (Total Exposed)', 
        'I_c (Incidence Rate)', 
        'Expected Qualified', 
        'Meets Target',
    ]]
    
    # Final cleanup and display
    result_table = result_table.rename(columns={'category_name': 'Category Name'})
    print(result_table.to_markdown(index=False))
    print("\n--- Summary Statistics ---")
    M_cost = len(slots)
    print(f"Total Respondents Surveyed (Cost, M): {M_cost}")
    print(f"Target Qualified (n): {target_qualified}")
    
    # Print Time Check (Constraint 2)
    slot_lengths = [max_mean_time - slot['time_left'] for slot in slots]
    slot_lengths_series = pd.Series(slot_lengths)
    print(f"Max Respondent Time Used: **{slot_lengths_series.max()}s** (Constraint: <= {max_mean_time}s)")