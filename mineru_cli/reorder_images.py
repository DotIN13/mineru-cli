def reorder_images(middle_json):
    """
    Reorders only the image-type blocks in `blocks` so that each image appears
    after the last block whose top (y0) coordinate is less than the image's y0.
    Other blocks remain in their original relative order. Updates each block's
    `index` to its new position.

    Args:
        blocks (List[dict]): List of blocks, each with keys 'type', 'bbox', and 'index'.
            bbox is a list [x0, y0, x1, y1].

    Returns:
        List[dict]: A new list of blocks with images re-ordered and indices updated.
    """
    middle_json = middle_json.copy()  # Avoid mutating input

    for page in middle_json.get("pdf_info", []):
        # Make a shallow copy to avoid mutating input
        reordered = list(page.get("para_blocks", []))

        i = 0
        while i < len(reordered):
            block = reordered[i]
            if block.get('type') == 'image':
                img_y0 = block['bbox'][1]
                # Determine the previous block's y0
                prev_y0 = reordered[i-1]['bbox'][1] if i > 0 else float('-inf')

                # If this image is "too high" (above its predecessor), move it earlier
                if img_y0 < prev_y0:
                    # Find the correct insertion position: just after the last block
                    # whose y0 is < img_y0
                    insert_pos = 0
                    for j, prev_block in enumerate(reordered[:i]):
                        if prev_block['bbox'][1] < img_y0:
                            insert_pos = j + 1

                    # Remove and insert at new position
                    reordered.pop(i)
                    reordered.insert(insert_pos, block)

                    # After moving, continue scanning from the insertion point
                    i = insert_pos + 1
                    continue  # skip the i += 1 at the bottom
            i += 1

        # Update indices to reflect new order
        for new_idx, blk in enumerate(reordered):
            blk['index'] = new_idx
        
        page["para_blocks"] = reordered

    return middle_json


if __name__ == "__main__":
    # Test it with a sample data/delivery/sample/working/mineru/NUSPEE/NUSPEE_07_questions/vlm/NUSPEE_07_questions_middle.json
    import json

    with open("/lambda/nfs/keensight/home/tzhang/tzhang3/test-papers/data/delivery/sample/working/mineru/NUSPEE/NUSPEE_07_questions/vlm/NUSPEE_07_questions_middle.json", "r") as f:
        data = json.load(f)

    new_middle_json = reorder_images(data)
    with open("/lambda/nfs/keensight/home/tzhang/tzhang3/test-papers/data/delivery/sample/working/mineru/NUSPEE/NUSPEE_07_questions/vlm/NUSPEE_07_questions_middle_reordered.json", "w") as out_f:
        json.dump(new_middle_json, out_f, indent=2, ensure_ascii=False)
    print(f"Reordered {len(new_middle_json.get('pdf_info', []))} pages, saved to NUSPEE_07_questions_middle_reordered.json")
