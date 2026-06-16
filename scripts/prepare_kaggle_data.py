import pandas as pd
import os
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help='Path to raw BugsRepo data dir')
    parser.add_argument('--out_file', type=str, required=True, help='Output merged CSV file')
    parser.add_argument('--nrows', type=int, default=None, help='Subset rows for testing')
    args = parser.parse_args()

    print(f"Loading data from {args.data_dir}...")
    
    # 1. Load Data
    meta = pd.read_csv(os.path.join(args.data_dir, 'Bug_meta_data.csv'), encoding='latin1', low_memory=False, nrows=args.nrows)
    contrib = pd.read_csv(os.path.join(args.data_dir, 'CSV_Contribution_information_dataset.csv'), encoding='latin1', nrows=args.nrows)
    comments = pd.read_csv(os.path.join(args.data_dir, 'comments_Dataset_Part_1.csv'), encoding='latin1', low_memory=False, nrows=args.nrows)

    # 2. Extract Reporter ID and Deduplicate Metadata
    print("Deduplicating and extracting reporter_id...")
    meta['reporter_id'] = meta['creator_detail'].str.extract(r"'id':\s*(\d+)")[0]
    meta['reporter_id'] = pd.to_numeric(meta['reporter_id'], errors='coerce')
    meta = meta.dropna(subset=['reporter_id', 'id']).drop_duplicates(subset=['id'])

    # 3. Deduplicate Contributor Data
    contrib['User ID'] = pd.to_numeric(contrib['User ID'], errors='coerce')
    contrib = contrib.dropna(subset=['User ID']).drop_duplicates(subset=['User ID'])

    # 4. Extract True Bug Description from Comments (First comment chronologically)
    print("Extracting first comments...")
    # Assuming comments are somewhat ordered or we just take the first one
    first_comments = comments.drop_duplicates(subset=['bug_id'], keep='first')

    # 5. Join Metadata with Contributor Features
    print("Joining metadata and contributor features...")
    merged = pd.merge(meta, contrib, left_on='reporter_id', right_on='User ID', how='inner')

    # 6. Join with Text Description
    print("Joining with text description...")
    final_df = pd.merge(merged, first_comments[['bug_id', 'text']], left_on='id', right_on='bug_id', how='inner')

    print(f"Final merged dataset has {len(final_df)} rows.")
    
    # Save to CSV
    final_df.to_csv(args.out_file, index=False)
    print(f"Saved to {args.out_file}")

if __name__ == '__main__':
    main()
