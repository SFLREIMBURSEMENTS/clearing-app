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

# ==============================================================================
# Step 2: All Helper Functions (Our Matching Engine)
# ==============================================================================
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
        return pd.Series([None, None, None, 0, 0, 0, 0])
    name_matches = process.extract(extracted_name, cust_choices, scorer=super_scorer, limit=5)
    if not name_matches:
        return pd.Series([None, None, None, 0, 0, 0, 0])
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
    other_candidates_list = []
    score_to_beat = best_match['final_score'] - CANDIDATE_SCORE_RANGE
    if len(sorted_candidates) > 1:
        for candidate in sorted_candidates[1:]:
             if candidate['final_score'] >= score_to_beat:
                 other_candidates_list.append(f"{candidate['ledger_name']} (Score: {candidate['final_score']:.2f})")
    other_candidates = "; ".join(other_candidates_list) if other_candidates_list else None
    return pd.Series([
        best_match['ledger_name'], other_candidates,
        best_match['final_score'], best_match['emi_count'],
        best_match['name_score'], best_match['amount_score']
    ])

@st.cache_data
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

# ==============================================================================
# Step 3: The Streamlit UI (The "Website" part)
# ==============================================================================
st.set_page_config(layout="wide")
st.title("🏦 Bank Transaction Matching Tool")

# --- Initialize session state ---
if 'matched_data' not in st.session_state:
    st.session_state.matched_data = None
if 'final_selections' not in st.session_state:
    st.session_state.final_selections = None

# --- Step 1: File Upload ---
st.header("Step 1: Upload Your Files")
col1, col2 = st.columns(2)
with col1:
    customer_file = st.file_uploader("Upload Customer List (with 'ledger_name')", type="csv")
with col2:
    bank_file = st.file_uploader("Upload Bank Statement (with 'narration', 'amount')", type="csv")

# --- Step 2: Processing ---
if customer_file and bank_file:
    if st.button("Start Matching Process", type="primary"):
        with st.spinner("Processing... This may take a moment."):
            # 1. Load data
            customers = pd.read_csv(customer_file)
            transactions = pd.read_csv(bank_file)

            # 2. Automated Data Splitting
            parser_regex = r'^(.*?)\s+(\d+\.?\d*)\s+([\w-]+)$'
            customers[['customer_name', 'emi_amount', 'customer_id']] = customers['ledger_name'].str.extract(parser_regex)

            # 3. Data Cleaning
            customers['emi_amount'] = pd.to_numeric(customers['emi_amount'], errors='coerce').fillna(0)
            transactions['amount'] = pd.to_numeric(transactions['amount'], errors='coerce').fillna(0)
            
            # 4. Text Preprocessing
            customers['clean_name'] = customers['customer_name'].apply(clean_text)
            transactions['extracted_name'] = transactions['narration'].apply(intelligent_name_extraction)
            customer_choices = customers['clean_name'].tolist()

            # 5. Run the matching (using .apply for Streamlit compatibility)
            results_df = transactions.apply(
                find_nearest_match, 
                args=(customers, customer_choices), 
                axis=1
            )
            results_df.columns = ['Matched Ledger Name', 'Other Candidates', 'Match Score', 'EMI Count', 'Name Score', 'Amount Score']
            
            # 6. Combine and prepare for display
            final_df = pd.concat([transactions, results_df], axis=1)
            
            # 7. Create columns for the interactive editor
            final_df['Selected Match'] = final_df['Matched Ledger Name']
            
            def create_options(row):
                options = []
                if pd.notna(row['Matched Ledger Name']):
                    options.append(row['Matched Ledger Name'])
                if pd.notna(row['Other Candidates']):
                    for item in row['Other Candidates'].split('; '):
                        options.append(item)
                return list(dict.fromkeys(options)) # Remove duplicates

            final_df['Selection Options'] = final_df.apply(create_options, axis=1)
            st.session_state.matched_data = final_df
        
        st.success("✅ Matching Complete! Please review the selections below.")

# --- Step 3: Interactive Review Table ---
if st.session_state.matched_data is not None:
    st.header("Step 2: Review and Make Selections")
    st.info("Click any cell in the 'Selected Match' column to choose the main match or an alternative.")

    df_to_edit = st.session_state.matched_data
    
    # Use st.data_editor to create the interactive table
    edited_df = st.data_editor(
        df_to_edit,
        column_config={
            # This makes the 'Selected Match' column a dropdown
            "Selected Match": st.column_config.SelectboxColumn(
                "Selected Match",
                help="Click to select the correct match",
                options=df_to_edit['Selection Options'],
                width="large"
            ),
            # Hide the helper columns
            "Selection Options": None,
            "extracted_name": None,
            "Matched Ledger Name": None,
            "EMI Count": None,
            "Name Score": None,
            "Amount Score": None
        },
        column_order=[
            "narration", "amount", "Selected Match", "Match Score", "Other Candidates"
        ],
        hide_index=True,
        use_container_width=True
    )
    
    # Store the user's final choices
    st.session_state.final_selections = edited_df

# --- Step 4: Download Button ---
if st.session_state.final_selections is not None:
    st.header("Step 3: Download Your Final Report")
    
    # Prepare final clean CSV (only the columns that matter)
    final_output_df = st.session_state.final_selections[
        ['narration', 'amount', 'Selected Match', 'Match Score', 'Other Candidates']
    ]
    
    csv_data = convert_df_to_csv(final_output_df)
    
    st.download_button(
        label="Download Final CSV",
        data=csv_data,
        file_name="final_matched_transactions.csv",
        mime="text/csv",
        type="primary"
    )