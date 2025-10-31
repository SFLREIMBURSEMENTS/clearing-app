import streamlit as st
import pandas as pd
from rapidfuzz import process, fuzz
import re
import io

# ==============================================================================
# Step 1: Configuration
# ==============================================================================
NAME_WEIGHT = 0.90
AMOUNT_WEIGHT = 0.10
AMOUNT_TOLERANCE_LOWER = 0.05
AMOUNT_TOLERANCE_UPPER = 0.20
FIRST_NAME_INTEGRITY_PENALTY = 0.75
MIDDLE_INITIAL_MISMATCH_PENALTY = 0.70
MIDDLE_NAME_MISMATCH_PENALTY = 0.75
UPI_MATCH_PENALTY = 0.75
CANDIDATE_SCORE_RANGE = 15
NONE_OPTION_TEXT = "-- NONE OF THE ABOVE --"

# ==============================================================================
# Step 2: All Helper Functions
# ==============================================================================
# ... (All helper functions remain exactly the same as the previous version) ...
def clean_text(text):
    if not isinstance(text, str): return ''
    text = text.lower().strip()
    text = re.sub(r'(bhai|sinh|kumar|mohd|mohammed|ben)', '', text, flags=re.IGNORECASE)
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def intelligent_name_extraction(narration):
    if not isinstance(narration, str): return ''
    narration_lower = narration.lower()
    if "upi" in narration_lower[:15]:
        upi_ids = re.findall(r'[\w.-]+@[\w.-]+', narration_lower)
        if not upi_ids: return ""
        name_part = upi_ids[0].split('@')[0]
        if name_part.isnumeric(): return ""
        cleaned_name = re.sub(r'[^a-z\s]', ' ', name_part).strip()
        cleaned_name = re.sub(r'\s+', ' ', cleaned_name)
        return cleaned_name
    else:
        text = narration_lower
        text = re.sub(r'(bhai|sinh|kumar|mohd|mohammed|ben)', ' ', text, flags=re.IGNORECASE)
        noise_patterns = [
            r'csh dep:[a-z]+', r'deposit by', r'cash deposit',
            r'\b(csh|dep|cash|by|tfr|frm|trf|fr|internal|account)\b',
            r'\d{10,}'
        ]
        for pattern in noise_patterns:
            text = re.sub(pattern, ' ', text, flags=re.IGNORECASE)
        text = re.sub(r'[^a-z\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

def super_scorer(s1, s2, **kwargs):
    if not s1 or not s2: return 0
    s1_words, s2_words = s1.split(), s2.split()
    if s1 == s2: return 100.0
    if len(s1_words) >= 2 and len(s2_words) > len(s1_words) and s2.startswith(s1): return 98.0
    if fuzz.token_set_ratio(s1, s2) == 100: return 97.0
    if len(s1_words) == 3 and len(s2_words) == 3 and len(s1_words[1]) == 1:
        if s1_words[0] == s2_words[0] and s1_words[2] == s2_words[2] and s1_words[1][0] == s2_words[1][0]:
            return 96.0
    base_score = fuzz.token_set_ratio(s1, s2)
    if len(s1_words) == 3 and len(s1_words[1]) == 1 and len(s2_words) == 3:
        if s1_words[1][0] != s2_words[1][0]: return base_score * MIDDLE_INITIAL_MISMATCH_PENALTY
    if len(s1_words) >= 3 and len(s2_words) >= 3:
        if s1_words[0] == s2_words[0] and s1_words[-1] == s2_words[-1] and s1_words[1] != s2_words[1]:
            return base_score * MIDDLE_NAME_MISMATCH_PENALTY
    if s1_words and s2_words and s2_words[0] not in s1_words:
         return base_score * FIRST_NAME_INTEGRITY_PENALTY
    return base_score

def get_amount_score(transaction_amt, emi_amt):
    best_score, best_multiplier = 0, 0
    if pd.isna(emi_amt) or emi_amt == 0: return 0, 0
    for multiplier in range(1, 5):
        base_emi = float(emi_amt) * multiplier
        lower_bound = base_emi * (1 - AMOUNT_TOLERANCE_LOWER)
        upper_bound = base_emi * (1 + AMOUNT_TOLERANCE_UPPER)
        if lower_bound <= float(transaction_amt) <= upper_bound:
            deviation = abs(float(transaction_amt) - base_emi) / base_emi if base_emi != 0 else 0
            score = 100 - (deviation * 100)
            if score > best_score:
                best_score, best_multiplier = score, multiplier
    return max(0, best_score), best_multiplier

def find_nearest_match(txn_row, cust_df, cust_choices):
    extracted_name = txn_row['extracted_name']
    txn_amount = txn_row['amount']
    narration = txn_row['narration']
    is_upi_txn = "upi" in str(narration)[:15].lower()
    if not extracted_name or not cust_choices:
        return pd.Series([None, None, 0, 0, 0, 0])
    name_matches = process.extract(extracted_name, cust_choices, scorer=super_scorer, limit=5)
    if not name_matches:
        return pd.Series([None, None, 0, 0, 0, 0])
    candidates = []
    for _matched_name, name_score, index in name_matches:
        customer_info = cust_df.iloc[index]
        amount_score, multiplier = get_amount_score(txn_amount, customer_info['emi_amount'])
        final_score = (name_score * NAME_WEIGHT) + (amount_score * AMOUNT_WEIGHT)
        if amount_score == 0: final_score = name_score * NAME_WEIGHT
        if is_upi_txn: final_score *= UPI_MATCH_PENALTY
        candidate_info = {
            'customer_id': customer_info['customer_id'],
            'ledger_name': customer_info['ledger_name'],
            'final_score': final_score, 'emi_count': multiplier if amount_score > 0 else 0,
            'name_score': name_score, 'amount_score': amount_score
        }
        candidates.append(candidate_info)
    sorted_candidates = sorted(candidates, key=lambda x: x['final_score'], reverse=True)
    best_match = sorted_candidates[0]
    
    # Build list of options for the UI
    options = []
    if pd.notna(best_match['ledger_name']):
        options.append(best_match['ledger_name'])
    score_to_beat = best_match['final_score'] - CANDIDATE_SCORE_RANGE
    if len(sorted_candidates) > 1:
        for candidate in sorted_candidates[1:]:
             if candidate['final_score'] >= score_to_beat:
                 options.append(f"{candidate['ledger_name']} (Score: {candidate['final_score']:.2f})")
    options = list(dict.fromkeys(options)) # remove duplicates
    options.append(NONE_OPTION_TEXT)

    return pd.Series([
        best_match['ledger_name'], options,
        best_match['final_score'], best_match['emi_count'],
        best_match['name_score'], best_match['amount_score']
    ])

@st.cache_data
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

# ==============================================================================
# Step 3: The Streamlit UI (with Click-to-Edit Panel)
# ==============================================================================
st.set_page_config(layout="wide")
st.title("🏦 Bank Transaction Matching Tool")

# --- Initialize session state ---
if 'matched_data' not in st.session_state:
    st.session_state.matched_data = None
# This state key will store the selection from the data_editor
if 'editor_key' not in st.session_state:
    st.session_state.editor_key = "data_editor_0"

# --- File Upload ---
st.header("Step 1: Upload Your Files")
col1, col2 = st.columns(2)
with col1:
    customer_file = st.file_uploader("Upload Customer List (with 'ledger_name')", type="csv")
with col2:
    bank_file = st.file_uploader("Upload Bank Statement (with 'narration', 'amount')", type="csv")

# --- Processing Button ---
if customer_file and bank_file:
    if st.button("Start Matching Process", type="primary"):
        with st.spinner("Processing... This may take a moment."):
            customers = pd.read_csv(customer_file, encoding='utf-8-sig')
            transactions = pd.read_csv(bank_file, encoding='utf-8-sig')
            if 'ledger_name' not in customers.columns:
                if len(customers.columns) > 0:
                    first_col_name = customers.columns[0]
                    st.toast(f"Warning: 'ledger_name' column not found. Using first column '{first_col_name}' instead.")
                    customers.rename(columns={first_col_name: 'ledger_name'}, inplace=True)
                else:
                    st.error("Error: The customer file is empty or has no columns.")
                    st.stop()
            parser_regex = r'^(.*?)\s+(\d+\.?\d*)\s+([\w-]+)$'
            customers[['customer_name', 'emi_amount', 'customer_id']] = customers['ledger_name'].str.extract(parser_regex)
            customers['emi_amount'] = pd.to_numeric(customers['emi_amount'], errors='coerce').fillna(0)
            transactions['amount'] = pd.to_numeric(transactions['amount'], errors='coerce').fillna(0)
            customers['clean_name'] = customers['customer_name'].apply(clean_text)
            transactions['extracted_name'] = transactions['narration'].apply(intelligent_name_extraction)
            customer_choices = customers['clean_name'].tolist()
            results_df = transactions.apply(
                find_nearest_match,
                args=(customers, customer_choices),
                axis=1
            )
            # 'Selection Options' is now returned from the match function
            results_df.columns = ['Matched Ledger Name', 'Selection Options', 'Match Score', 'EMI Count', 'Name Score', 'Amount Score']
            final_df = pd.concat([transactions, results_df], axis=1)
            final_df['Selected Match'] = final_df['Matched Ledger Name']
            
            # Reset the editor key to force a re-render
            st.session_state.editor_key = f"data_editor_{pd.Timestamp.now().isoformat()}"
            st.session_state.matched_data = final_df
        
        st.success("✅ Matching Complete! Please review the selections below.")


# --- Display the Data Table ---
if st.session_state.matched_data is not None:
    st.header("Step 2: Review and Make Selections")
    st.info("Click a row in the table below to select it. An edit panel will appear beneath the table.")
    
    # We'll use a copy to display in the editor
    df_to_display = st.session_state.matched_data.copy()

    # Use st.data_editor. This is our main table.
    edited_df = st.data_editor(
        df_to_display,
        key=st.session_state.editor_key, # Use the dynamic key
        column_config={
            # We make "Selected Match" read-only here
            "Selected Match": st.column_config.TextColumn(
                "Selected Match",
                width="large"
            ),
            # Hide helper columns
            "Selection Options": None, "extracted_name": None, "Matched Ledger Name": None,
            "EMI Count": None, "Name Score": None, "Amount Score": None
        },
        column_order=[
            "narration", "amount", "Selected Match", "Match Score"
        ],
        hide_index=True,
        use_container_width=True,
        on_select="rerun" # Tell Streamlit to rerun when a selection changes
    )

    # --- Step 3: The "Click-to-Edit" Panel ---
    selection = st.session_state[st.session_state.editor_key].get("selection")
    
    if selection and selection["rows"]:
        # Get the index of the last row the user clicked on
        selected_index = selection["rows"][-1] 
        selected_row = st.session_state.matched_data.iloc[selected_index]

        # This container appears below the table
        with st.container(border=True):
            st.subheader(f"Edit Selection for:")
            st.markdown(f"**Narration:** {selected_row['narration']} | **Amount:** {selected_row['amount']}")
            
            options = selected_row['Selection Options']
            current_selection = selected_row['Selected Match']
            
            # Find the index of the current selection
            if pd.isna(current_selection):
                current_selection_string = NONE_OPTION_TEXT
            else:
                current_selection_string = current_selection

            try:
                current_index = options.index(current_selection_string)
            except ValueError:
                # Handle cases where the selection is not in the options list (e.g., old edit)
                # Add it to the list and select it
                if current_selection_string != NONE_OPTION_TEXT:
                    options.insert(0, current_selection_string)
                current_index = 0

            # This radio button writes the change directly to the session state
            new_selection = st.radio(
                "Choose the correct match:",
                options,
                index=current_index,
                key=f"radio_{selected_index}",
                horizontal=True
            )

            # This 'on_change' callback is the key to performance
            # It updates the main DataFrame *without* a full page rerun
            if new_selection != current_selection_string:
                if new_selection == NONE_OPTION_TEXT:
                    final_selection_name = None
                elif "(Score:" in new_selection:
                    final_selection_name = new_selection.split(" (Score:")[0]
                else:
                    final_selection_name = new_selection
                
                # Update the master dataframe in session state
                st.session_state.matched_data.loc[selected_index, 'Selected Match'] = final_selection_name
                
                # Rerun to update the table display with the new selection
                st.rerun()


# --- Download Button ---
if st.session_state.matched_data is not None:
    st.header("Step 4: Download Your Final Report")
    
    # The dataframe in session_state always has the latest edits
    final_output_df = st.session_state.matched_data[
        ['narration', 'amount', 'Selected Match', 'Match Score']
    ]
    
    csv_data = convert_df_to_csv(final_output_df)
    
    st.download_button(
        label="Download Final CSV",
        data=csv_data,
        file_name="final_matched_transactions.csv",
        mime="text/csv",
        type="primary"
    )