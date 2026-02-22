def download_all_then_save_first_200(output_file="pinecone_first_200_sorted.csv"):
    """
    Download ALL records from Pinecone, sort by Doc_Title and Chunk_Index,
    then save the first 200
    """
    # Get API keys
    pinecone_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_MEDLINE_TEST_INDEX_NAME")

    if not pinecone_key or not index_name:
        raise ValueError("❌ Missing PINECONE_API_KEY or index name")

    # Initialize Pinecone
    print(f"Connecting to Pinecone index: {index_name}")
    pc = Pinecone(api_key=pinecone_key)
    index = pc.Index(index_name)

    # Get index stats
    stats = index.describe_index_stats()
    total_vectors = stats.get('total_vector_count', 0)
    print(f"Total vectors in index: {total_vectors}")

    # Step 1: Get ALL IDs from Pinecone
    print(f"\nFetching ALL IDs from index...")
    all_ids = []
    pagination_token = None
    page_limit = 100  # Maximum allowed by Pinecone per page
    page_count = 0

    while True:
        try:
            page_count += 1
            print(f"  Fetching page {page_count} (total IDs so far: {len(all_ids)})...")

            # List with pagination
            if pagination_token:
                response = index.list_paginated(limit=page_limit, pagination_token=pagination_token)
            else:
                response = index.list_paginated(limit=page_limit)

            # Extract IDs from response
            if hasattr(response, 'vectors'):
                ids_batch = [v.id for v in response.vectors]
            elif isinstance(response, list):
                ids_batch = response
            else:
                ids_batch = list(response)

            all_ids.extend(ids_batch)
            print(f"    Retrieved {len(ids_batch)} IDs (total: {len(all_ids)})")

            # Check if there are more pages
            if hasattr(response, 'pagination') and response.pagination:
                pagination_token = response.pagination.next
                if not pagination_token:
                    print("  ✓ No more pages available")
                    break
            else:
                print("  ✓ Reached end of index")
                break

        except Exception as e:
            print(f"  Error during pagination: {e}")
            break

    print(f"\n✓ Retrieved {len(all_ids)} total IDs from index")

    if not all_ids:
        print("❌ No records found in index")
        return

    # Step 2: Fetch ALL records in batches
    batch_size = 100
    all_records = []

    print(f"\nFetching ALL record details in batches of {batch_size}...")
    total_batches = (len(all_ids) - 1) // batch_size + 1

    for i in range(0, len(all_ids), batch_size):
        batch_ids = all_ids[i:i + batch_size]
        batch_num = i // batch_size + 1
        print(f"  Batch {batch_num}/{total_batches}: Fetching {len(batch_ids)} records...")

        try:
            fetch_response = index.fetch(ids=batch_ids)

            for vector_id, vector_data in fetch_response['vectors'].items():
                metadata = vector_data.get('metadata', {})

                record = {
                    'pinecone_id': vector_id,
                    'doc_title': metadata.get('Doc_Title', ''),
                    'sec_title': metadata.get('Sec_Title', ''),
                    'splitted': metadata.get('Splitted', ''),
                    'chunk_index': metadata.get('Chunk_Index', 0),
                    'page_content': metadata.get('text', '')
                }
                all_records.append(record)

        except Exception as e:
            print(f"    Error fetching batch: {e}")
            continue

    print(f"\n✓ Retrieved {len(all_records)} total records")

    # Step 3: Sort ALL records by Doc_Title, Sec_Title, then Chunk_Index
    print(f"\nSorting ALL {len(all_records)} records by document and chunk order...")
    all_records_sorted = sorted(
        all_records,
        key=lambda x: (
            x['doc_title'],  # First by document title
            x['sec_title'],  # Then by section title
            int(x['chunk_index']) if isinstance(x['chunk_index'], (int, str)) and str(x['chunk_index']).isdigit() else 0
        # Then by chunk index
        )
    )
    print("✓ Sorting complete")

    # Step 4: Take only first 200 sorted records  ← CHANGED FROM 100 TO 200
    first_200_records = all_records_sorted[:200]
    print(f"\n✓ Selected first 200 records after sorting")

    # Step 5: Write to CSV
    print(f"\nWriting to {output_file}...")
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['row_num', 'pinecone_id', 'doc_title', 'sec_title', 'splitted',
                      'chunk_index', 'page_content']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for i, record in enumerate(first_200_records, 1):
            record['row_num'] = i  # Add row number
            writer.writerow(record)

    print(f"✓ Successfully saved 200 records to {output_file}")

    # Step 6: Show summary statistics
    print("\n" + "=" * 60)
    print("Summary of ALL records in index:")
    print("=" * 60)

    # Count unique documents
    unique_docs = set(r['doc_title'] for r in all_records_sorted)
    print(f"Total records in index: {len(all_records_sorted)}")
    print(f"Unique documents: {len(unique_docs)}")

    # Show first 200 document distribution  ← CHANGED FROM 100 TO 200
    print("\n" + "=" * 60)
    print("First 200 records breakdown:")
    print("=" * 60)

    current_doc = None
    doc_count = 0
    chunk_count = 0

    for record in first_200_records:  # ← CHANGED FROM first_100_records
        if record['doc_title'] != current_doc:
            if current_doc is not None:
                print(f"  └─ Chunks: {chunk_count}")
            current_doc = record['doc_title']
            doc_count += 1
            chunk_count = 0
            print(f"\n{doc_count}. {current_doc[:60]}{'...' if len(current_doc) > 60 else ''}")
        chunk_count += 1

    if current_doc is not None:
        print(f"  └─ Chunks: {chunk_count}")

    print(f"\nDocuments in first 200: {doc_count}")  # ← CHANGED FROM 100 TO 200

    # Show first 5 records
    print("\n" + "=" * 60)
    print("First 5 records (from the 200 saved):")  # ← CHANGED FROM 100 TO 200
    print("=" * 60)
    for i, record in enumerate(first_200_records[:5], 1):  # ← CHANGED FROM first_100_records
        print(f"\n{i}. Doc: {record['doc_title']}")
        print(f"   Section: {record['sec_title']}")
        print(f"   Chunk: {record['chunk_index']} | Splitted: {record['splitted']}")
        content = record['page_content'][:80] if record['page_content'] else "(No content)"
        print(f"   Content: {content}...")

    return first_200_records  # ← CHANGED FROM first_100_records

