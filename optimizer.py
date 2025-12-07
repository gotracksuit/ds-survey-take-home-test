import numpy as np
from scipy.optimize import linprog
import pandas as pd


def lp_allocation(categories,
                     demographics,
                     demographic_shares,
                     target_qualified,
                     max_mean_time,
                     demographic_tolerance=0.05):
    """
    LP with relaxed demographic constraints.

    Parameters
    ----------
    categories : pd.DataFrame
        Must include incidence_rate and category_length_seconds
    demographics : list
        Demographic cell names
    demographic_shares : np.ndarray
        National population shares for each cell
    target_qualified : int
        Target qualified completes per category
    max_mean_time : float
        Maximum allowed mean survey time (seconds)
    demographic_tolerance : float
        Allowed deviation in demographic shares (e.g., 0.05 for ±5%)    

    Returns
    -------
    np.ndarray
        Shape (num_categories, num_demographics)    
    """

    num_categories = len(categories)
    num_demographics = len(demographics)

    def idx(i, j):
        return i * num_demographics + j

    num_vars = num_categories * num_demographics

    # Objective: minimize total respondents
    c = np.ones(num_vars)

    A = []
    b = []

    # Qualified completes constraint (HARD)
    for i in range(num_categories):
        row = np.zeros(num_vars)
        for j in range(num_demographics):
            row[idx(i, j)] = categories.loc[i, "incidence_rate"]
        A.append(-row)
        b.append(-target_qualified)

    # Relaxed demographic constraints (GLOBAL)
    # sum_i x_ij ∈ [ (p_j - eps) * N , (p_j + eps) * N ]
    # where N = sum_{i,j} x_ij

    for j, p_j in enumerate(demographic_shares):
        lower_row = np.zeros(num_vars)
        upper_row = np.zeros(num_vars)

        for i in range(num_categories):
            lower_row[idx(i, j)] = -1
            upper_row[idx(i, j)] = 1

        for k in range(num_vars):
            lower_row[k] += (p_j - demographic_tolerance)
            upper_row[k] -= (p_j + demographic_tolerance)

        # lower bound: sum_i x_ij ≥ (p_j - eps) * N
        A.append(lower_row)
        b.append(0)

        # upper bound: sum_i x_ij ≤ (p_j + eps) * N
        A.append(upper_row)
        b.append(0)

    # Mean survey length: expected time ≤ 480s
    time_row = np.zeros(num_vars)
    count_row = np.ones(num_vars)

    for i in range(num_categories):
        p = categories.loc[i, "incidence_rate"]
        t = categories.loc[i, "category_length_seconds"]
        for j in range(num_demographics):
            time_row[idx(i, j)] = p * t

    A.append(time_row - max_mean_time * count_row)
    b.append(0)

    result = linprog(
        c=c,
        A_ub=A,
        b_ub=b,
        bounds=(0, None),
        method="highs"
    )

    if not result.success:
        raise RuntimeError(result.message)

    print(f"✅ LP solved: total respondents = {result.fun:.1f}")

    return result.x.reshape(num_categories, num_demographics)


class CategoryAllocatorGreedy():
    '''
    Multi-quota greedy category allocator.
    Parameters
    ----------  

    df_categories : pd.DataFrame
        Must include category_id, incidence_rate, category_length_seconds
    demographics : pd.DataFrame
        Must include cell_id, population_share
    target_qualified : int 
        Target qualified completes per category
    max_time : float
        Maximum allowed interview time per respondent (seconds)
    ----------
    Methods
    -------
    run_allocation()
        Executes the greedy allocation algorithm.
    Returns
    -------
    list of dict
        Each dict represents a respondent slot with assigned categories and remaining time. 
    
    '''
    def __init__(self,
                 df_categories,
                 demographics,
                 target_qualified,
                 max_time):
        
        df_categories['required_exposure'] = np.ceil(target_qualified / df_categories['incidence_rate']).astype(int)
        df_categories['efficiency_score'] = df_categories['required_exposure'] / df_categories['category_length_seconds']
        # Sort by efficiency descending (Greedy choice)
        df_categories = df_categories.sort_values(by='efficiency_score', ascending=False).reset_index(drop=True)
        self.categories_df = df_categories

        self.max_time = max_time
        self.demo_shares = dict(zip(demographics['cell_id'], demographics['population_share']))
        self.required_remaining = df_categories.set_index('category_id')['required_exposure'].copy()
        self.exposure_counts = {cid: 0 for cid in df_categories['category_id']}
        self.respondent_slots_by_demo = {cell: [] for cell in self.demo_shares.keys()}
        self.total_respondents_created = 0

    def get_most_needed_category(self):
        """Selects the category with the highest remaining exposure and efficiency."""

        active_df = self.categories_df.copy()
        active_df['remaining'] = active_df['category_id'].map(self.required_remaining)
        active_df = active_df[active_df['remaining'] > 0]
        
        if active_df.empty: 
            return None

        c_star_data = active_df.sort_values(
            by=['remaining', 'efficiency_score'], 
            ascending=[False, False]
        ).iloc[0]
        
        return c_star_data['category_id'], c_star_data['category_length_seconds']

    def get_most_needed_demographic(self):
        """Selects the demographic cell currently most underrepresented."""

        actual_dist = {cell: len(slots) for cell, slots in self.respondent_slots_by_demo.items()}
        if self.total_respondents_created == 0:
            return max(self.demo_shares, key=self.demo_shares.get)

        expected_count = self.total_respondents_created * pd.Series(self.demo_shares)
        actual_count = pd.Series(actual_dist)
        under_represented = expected_count - actual_count
        
        return under_represented.idxmax()

    def run_allocation(self):
        """Main loop for the multi-quota greedy allocation process."""

        while True:
            result = self.get_most_needed_category()
            if result is None: 
                break
            
            c_star_id, c_star_length = result
            j_star_cell = self.get_most_needed_demographic()
            slots = self.respondent_slots_by_demo[j_star_cell]
            
            # Find Best-Fit slot
            best_slot_index = -1
            min_time_left = self.max_time + 1
            
            for i, slot in enumerate(slots):
                if c_star_length <= slot['time_left'] and c_star_id not in slot['categories']:
                    if slot['time_left'] < min_time_left:
                        min_time_left = slot['time_left']
                        best_slot_index = i
                        
            # Assign or Create
            if best_slot_index != -1:
                slot = slots[best_slot_index]
            else:
                self.total_respondents_created += 1
                slot = {'id': self.total_respondents_created, 'time_left': self.max_time, 'categories': [], 'demographic_cell': j_star_cell}
                slots.append(slot)
            
            # Perform the assignment
            slot['categories'].append(c_star_id)
            slot['time_left'] -= c_star_length
            
            self.required_remaining[c_star_id] -= 1
            self.exposure_counts[c_star_id] += 1
            
        final_slots = [slot for slots in self.respondent_slots_by_demo.values() for slot in slots]
        return final_slots
